import reflex as rx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.getcwd())
try:
    from chatbot.rag_engine import RAGEngine
    global_rag_engine = RAGEngine()
except ImportError:
    global_rag_engine = None

# ==========================================
# 🎨 DESIGN SYSTEM & THEME TOKENS
# ==========================================
BG_COLOR = "#0F172A" # Slate 900
CARD_BG = "rgba(30, 41, 59, 0.7)" # Slate 800 with opacity for glass effect
ACCENT = "#6366F1" # Indigo 500
TEXT_MAIN = "#F8FAFC"
TEXT_SUB = "#94A3B8"
BORDER_COLOR = "rgba(255, 255, 255, 0.1)"

def glass_card(*children, **props):
    """Premium glassmorphism card component."""
    final_props = {
        "bg": CARD_BG,
        "backdrop_filter": "blur(12px)",
        "border": f"1px solid {BORDER_COLOR}",
        "border_radius": "16px",
        "box_shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
        "padding": "1.5em",
        "width": "100%",
        "overflow": "hidden",
    }
    final_props.update(props)
    return rx.box(
        *children,
        **final_props
    )

def tier_badge(clv_value) -> rx.Component:
    """Returns a styled Gold/Silver/Bronze badge based on CLV."""
    clv_val = clv_value.to(float)
    return rx.cond(
        clv_val > 2000,
        rx.badge("🥇 Gold Tier", color_scheme="yellow", radius="full", size="2"),
        rx.cond(
            clv_val > 500,
            rx.badge("🥈 Silver Tier", color_scheme="gray", radius="full", size="2"),
            rx.badge("🥉 Bronze Tier", color_scheme="orange", radius="full", size="2")
        )
    )

# ==========================================
# 🧠 GLOBAL STATE
# ==========================================
class AppState(rx.State):
    """The global application state."""
    
    # KPIs
    total_revenue: str = "$0"
    avg_clv: str = "$0"
    high_risk_count: str = "0"
    active_customers: str = "0"
    
    # Data Tables
    top_customers: list[dict] = []
    at_risk_customers: list[dict] = []
    
    # Interactive Plotly Figures
    fig_segmentation: go.Figure = go.Figure()
    fig_rfm: go.Figure = go.Figure()
    fig_churn: go.Figure = go.Figure()
    fig_forecast: go.Figure = go.Figure()
    
    # Chatbot state
    chat_history: list[dict] = [
        {"role": "assistant", "content": "Hello! I am your EIDAP AI Assistant. Ask me anything about our customers, churn risks, or forecasts."}
    ]
    user_message: str = ""
    is_chatting: bool = False
    
    data_loaded: bool = False
    data_error: str = ""

    def set_user_message(self, val: str):
        self.user_message = val

    def load_data(self):
        try:
            if not os.path.exists("data/enterprise_intelligence.csv"):
                self.data_error = "Data files not found. Please run the Data & ML Pipeline (python run_pipelines.py)."
                self.data_loaded = False
                return

            df = pd.read_csv("data/enterprise_intelligence.csv")
            self.total_revenue = f"${df['monetary'].sum():,.0f}"
            self.avg_clv = f"${df['predicted_clv'].mean():,.0f}"
            self.high_risk_count = str(len(df[df['churn_risk'] == 'High']))
            self.active_customers = str(len(df[df['churned'] == 0]))
            
            # --- DATA TABLES ---
            # Top Customers by CLV
            top_df = df.nlargest(10, 'predicted_clv')
            self.top_customers = top_df[['customer_id', 'segment_label', 'monetary', 'predicted_clv']].to_dict('records')
            
            # At Risk Customers
            risk_df = df[df['churn_risk'] == 'High'].nlargest(10, 'monetary')
            self.at_risk_customers = risk_df[['customer_id', 'monetary', 'recency', 'next_best_action']].to_dict('records')
            
            # --- CHARTS ---
            # Segment Pie
            seg_counts = df['segment_label'].value_counts().reset_index()
            seg_counts.columns = ['Segment', 'Count']
            self.fig_segmentation = px.pie(seg_counts, values='Count', names='Segment', hole=0.5, template="plotly_dark", 
                                           color_discrete_sequence=px.colors.qualitative.Pastel)
            self.fig_segmentation.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=0, b=0, l=0, r=0))
            
            # 3D RFM
            sample_df = df.sample(min(500, len(df)))
            self.fig_rfm = px.scatter_3d(sample_df, x='recency', y='frequency', z='monetary', color='segment_label', 
                                         template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
            self.fig_rfm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=0, b=0, l=0, r=0))
            
            # Churn Bar Chart
            churn_counts = df['churn_risk'].value_counts().reset_index()
            churn_counts.columns = ['Risk', 'Count']
            self.fig_churn = px.bar(churn_counts, x='Risk', y='Count', color='Risk', 
                                    color_discrete_map={'High': '#EF4444', 'Medium': '#F59E0B', 'Low': '#10B981'}, template="plotly_dark")
            self.fig_churn.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=0, b=0, l=0, r=0))
            
            # Forecast Line Chart
            try:
                forecast = pd.read_csv('data/forecast_90d.csv')
                tx = pd.read_csv("data/transactions.csv")
                tx['date'] = pd.to_datetime(tx['date']).dt.date
                daily = tx.groupby('date')['amount'].sum().reset_index()
                daily.columns = ['ds', 'y']
                daily['ds'] = pd.to_datetime(daily['ds'])
                forecast['ds'] = pd.to_datetime(forecast['ds'])
                
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=daily['ds'], y=daily['y'], name="Actual Revenue", line=dict(color=ACCENT, width=2)))
                last_actual = daily['ds'].max()
                future = forecast[forecast['ds'] > last_actual]
                
                fig_fc.add_trace(go.Scatter(x=future['ds'], y=future['yhat'], name="Predicted", line=dict(color='#38BDF8', dash='dash', width=2)))
                fig_fc.add_trace(go.Scatter(x=future['ds'], y=future['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
                fig_fc.add_trace(go.Scatter(x=future['ds'], y=future['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', name='Confidence', fillcolor='rgba(56, 189, 248, 0.2)'))
                
                fig_fc.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=0, b=0, l=0, r=0))
                self.fig_forecast = fig_fc
            except Exception:
                pass
                
            self.data_loaded = True
            self.data_error = ""
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data_error = str(e)
            self.data_loaded = False
            
    async def handle_submit(self):
        if not self.user_message.strip():
            return
        
        self.chat_history.append({"role": "user", "content": self.user_message})
        current_msg = self.user_message
        self.user_message = ""
        self.is_chatting = True
        yield
        
        try:
            if global_rag_engine:
                response = global_rag_engine.query(current_msg)
            else:
                response = "RAG Engine is not initialized."
        except Exception as e:
            response = f"Error querying Assistant: {e}"
            
        self.chat_history.append({"role": "assistant", "content": response})
        self.is_chatting = False

# ==========================================
# 🏗️ UI COMPONENTS
# ==========================================

def sidebar_nav() -> rx.Component:
    """Modern sticky sidebar for navigation."""
    def nav_link(text: str, url: str, icon: str) -> rx.Component:
        return rx.link(
            rx.hstack(
                rx.icon(tag=icon, size=18, color=TEXT_SUB),
                rx.text(text, size="3", weight="medium", color=TEXT_SUB),
                spacing="3",
                align_items="center",
                padding_y="0.75em",
                padding_x="1em",
                border_radius="8px",
                _hover={"bg": "rgba(255,255,255,0.05)", "color": TEXT_MAIN}
            ),
            href=url,
            underline="none"
        )

    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.icon(tag="command", size=24, color=ACCENT),
                rx.heading("EIDAP", size="6", color=TEXT_MAIN, letter_spacing="-0.5px"),
                align_items="center",
                margin_bottom="2em",
                padding_x="1em"
            ),
            nav_link("Dashboard", "/", "layout-dashboard"),
            nav_link("Customer 360", "/customer", "users"),
            nav_link("Churn Intelligence", "/churn", "activity"),
            nav_link("Lifetime Value", "/clv", "gem"),
            nav_link("Forecasting", "/forecast", "trending-up"),
            nav_link("AI Assistant", "/assistant", "bot"),
            spacing="1",
            width="250px",
            height="100vh",
            position="fixed",
            left="0",
            top="0",
            bg=CARD_BG,
            border_right=f"1px solid {BORDER_COLOR}",
            padding_top="2em",
            padding_x="1em",
            display=["none", "none", "flex"]
        )
    )

def page_layout(*children, title: str, subtitle: str) -> rx.Component:
    """Wraps page content with sidebar and headers."""
    content_area = rx.cond(
        AppState.data_loaded,
        rx.vstack(
            rx.vstack(
                rx.heading(title, size="8", color=TEXT_MAIN, letter_spacing="-1px"),
                rx.text(subtitle, size="4", color=TEXT_SUB),
                spacing="1",
                margin_bottom="2em"
            ),
            *children,
            spacing="6",
            width="100%",
            max_width="1200px",
            margin="0 auto"
        ),
        rx.center(
            rx.vstack(
                rx.icon(tag="database-zap", size=64, color=ACCENT),
                rx.heading("Pipeline Not Run", size="8", color=TEXT_MAIN),
                rx.text("The required data files are missing.", size="4", color=TEXT_SUB),
                rx.text(AppState.data_error, color="#EF4444", margin_bottom="1em"),
                rx.code_block("python run_pipelines.py", language="bash", margin_bottom="1em"),
                rx.text("Run the command above or ensure the pipeline ran successfully before accessing the dashboard.", color=TEXT_SUB, size="2"),
                spacing="4",
                align_items="center",
                text_align="center",
                bg=CARD_BG,
                padding="3em",
                border_radius="16px",
                border=f"1px solid {BORDER_COLOR}",
                box_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1)"
            ),
            width="100%",
            min_height="80vh"
        )
    )

    return rx.box(
        sidebar_nav(),
        rx.box(
            content_area,
            margin_left=["0", "0", "250px"],
            padding=["1.5em", "2em", "3em"]
        ),
        bg=BG_COLOR,
        min_height="100vh",
        font_family="Inter, sans-serif"
    )

def kpi_card(title: str, value: str, icon: str, trend: str = None) -> rx.Component:
    """Glass-styled top KPI card."""
    return glass_card(
        rx.hstack(
            rx.vstack(
                rx.text(title, size="2", color=TEXT_SUB, font_weight="bold", text_transform="uppercase", letter_spacing="1px"),
                rx.heading(value, size="7", color=TEXT_MAIN),
                rx.cond(
                    trend != None,
                    rx.text(trend, size="2", color="#10B981" if "+" in str(trend) else "#EF4444"),
                    rx.box()
                ),
                spacing="1"
            ),
            rx.spacer(),
            rx.box(
                rx.icon(tag=icon, size=24, color=ACCENT),
                padding="0.75em",
                bg="rgba(99, 102, 241, 0.1)",
                border_radius="12px"
            ),
            width="100%",
            align_items="start"
        ),
        width="100%"
    )

# ==========================================
# 📄 PAGES
# ==========================================

def index() -> rx.Component:
    return page_layout(
        rx.grid(
            kpi_card("Total Revenue", AppState.total_revenue, "dollar-sign", trend="+12.5% YoY"),
            kpi_card("Average CLV", AppState.avg_clv, "trending-up", trend="+4.2% YoY"),
            kpi_card("Active Customers", AppState.active_customers, "users", trend="+1.1% MoM"),
            kpi_card("At-Risk Accounts", AppState.high_risk_count, "triangle-alert", trend="-2.0% MoM"),
            columns="4",
            spacing="4",
            width="100%"
        ),
        rx.grid(
            glass_card(
                rx.text("Customer Segmentation Profile", size="4", weight="bold", margin_bottom="1em"),
                rx.plotly(data=AppState.fig_segmentation, height="350px", width="100%")
            ),
            glass_card(
                rx.text("90-Day Revenue Trajectory", size="4", weight="bold", margin_bottom="1em"),
                rx.plotly(data=AppState.fig_forecast, height="350px", width="100%")
            ),
            columns="2",
            spacing="6",
            width="100%"
        ),
        title="Executive Dashboard",
        subtitle="High-level enterprise decision metrics and health overview."
    )

def customer_360() -> rx.Component:
    return page_layout(
        glass_card(
            rx.text("RFM Behavioral Clusters (3D)", size="4", weight="bold", margin_bottom="1em"),
            rx.plotly(data=AppState.fig_rfm, height="500px", width="100%")
        ),
        rx.grid(
            glass_card(
                rx.hstack(rx.icon("award", color="#F59E0B"), rx.text("Champions", weight="bold"), align="center"),
                rx.text("Highly engaged, high spenders. Ready for cross-sells.", color=TEXT_SUB, size="2", margin_top="0.5em")
            ),
            glass_card(
                rx.hstack(rx.icon("target", color="#38BDF8"), rx.text("Core Accounts", weight="bold"), align="center"),
                rx.text("Steady recurring revenue. Focus on retention.", color=TEXT_SUB, size="2", margin_top="0.5em")
            ),
            glass_card(
                rx.hstack(rx.icon("circle-alert", color="#EF4444"), rx.text("At-Risk", weight="bold"), align="center"),
                rx.text("Dropping frequency. Needs immediate win-back campaigns.", color=TEXT_SUB, size="2", margin_top="0.5em")
            ),
            columns="3",
            spacing="4"
        ),
        title="Customer 360",
        subtitle="Analyze segment health and behavioral clustering."
    )

def churn_intelligence() -> rx.Component:
    return page_layout(
        rx.grid(
            glass_card(
                rx.text("Churn Risk Distribution", size="4", weight="bold", margin_bottom="1em"),
                rx.plotly(data=AppState.fig_churn, height="350px")
            ),
            glass_card(
                rx.text("Critical Action Required: Top At-Risk", size="4", weight="bold", margin_bottom="1em"),
                rx.box(
                    rx.foreach(
                        AppState.at_risk_customers,
                        lambda row: rx.hstack(
                            rx.vstack(
                                rx.text(f"Customer #{row['customer_id']}", weight="bold"),
                                rx.text(f"Last active: {row['recency']} days ago", size="1", color=TEXT_SUB)
                            ),
                            rx.spacer(),
                            rx.badge(row['next_best_action'], color_scheme="red", radius="full"),
                            width="100%",
                            padding_y="0.75em",
                            border_bottom=f"1px solid {BORDER_COLOR}"
                        )
                    ),
                    overflow_y="auto",
                    height="350px"
                )
            ),
            columns="2",
            spacing="6"
        ),
        title="Churn Intelligence",
        subtitle="Predictive risk scoring and automated next-best-actions."
    )

def clv_leaderboard() -> rx.Component:
    return page_layout(
        glass_card(
            rx.text("Top Enterprise Accounts by Predicted CLV", size="4", weight="bold", margin_bottom="1em"),
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.table.column_header_cell("Customer ID"),
                        rx.table.column_header_cell("Segment"),
                        rx.table.column_header_cell("Historical Spend"),
                        rx.table.column_header_cell("Predicted 12M Value"),
                        rx.table.column_header_cell("Tier")
                    )
                ),
                rx.table.body(
                    rx.foreach(
                        AppState.top_customers,
                        lambda row: rx.table.row(
                            rx.table.cell(rx.text(row['customer_id'], weight="bold")),
                            rx.table.cell(row['segment_label']),
                            rx.table.cell(f"${row['monetary']}"),
                            rx.table.cell(rx.text(f"${row['predicted_clv']}", color="#10B981", weight="bold")),
                            rx.table.cell(tier_badge(row['predicted_clv']))
                        )
                    )
                ),
                width="100%"
            )
        ),
        title="Lifetime Value",
        subtitle="Identify and nurture the most valuable customers."
    )

def forecasting() -> rx.Component:
    return page_layout(
        glass_card(
            rx.text("Prophet ML Demand Forecasting", size="4", weight="bold", margin_bottom="1em"),
            rx.plotly(data=AppState.fig_forecast, height="600px", width="100%")
        ),
        title="Demand Forecasting",
        subtitle="Time-series prediction of upcoming 90-day revenue trends."
    )

def assistant() -> rx.Component:
    def chat_message(msg) -> rx.Component:
        is_user = msg["role"] == "user"
        return rx.hstack(
            rx.avatar(fallback=rx.cond(is_user, "U", "AI"), color_scheme=rx.cond(is_user, "blue", "indigo"), size="2"),
            rx.box(
                rx.markdown(msg["content"]),
                bg=rx.cond(is_user, ACCENT, "rgba(255,255,255,0.05)"),
                color="white",
                padding="1em",
                border_radius="16px",
                border_top_right_radius=rx.cond(is_user, "4px", "16px"),
                border_top_left_radius=rx.cond(is_user, "16px", "4px"),
                max_width="80%",
                box_shadow="md"
            ),
            align_items="start",
            justify_content=rx.cond(is_user, "end", "start"),
            width="100%",
            margin_bottom="1em"
        )

    return page_layout(
        glass_card(
            rx.box(
                rx.foreach(AppState.chat_history, chat_message),
                rx.cond(AppState.is_chatting, rx.spinner(color=ACCENT), rx.box()),
                height="500px",
                overflow_y="auto",
                padding_right="1em",
                margin_bottom="1em"
            ),
            rx.hstack(
                rx.input(
                    placeholder="Ask 'Why is churn high?' or 'Which segment is most profitable?'",
                    value=AppState.user_message,
                    on_change=AppState.set_user_message,
                    width="100%",
                    size="3",
                    radius="full",
                    bg="rgba(0,0,0,0.2)",
                    border=f"1px solid {BORDER_COLOR}"
                ),
                rx.button(
                    rx.icon("send", size=18),
                    on_click=AppState.handle_submit,
                    size="3",
                    radius="full",
                    bg=ACCENT,
                    cursor="pointer"
                ),
                width="100%"
            ),
            width="100%",
            max_width="800px",
            margin="0 auto"
        ),
        title="AI Business Assistant",
        subtitle="Talk to your data using the RAG-powered intelligence engine."
    )

# ==========================================
# 🚀 APP REGISTRATION
# ==========================================
app = rx.App(
    theme=rx.theme(appearance="dark", has_background=True, radius="large", accent_color="indigo"),
    stylesheets=["https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"]
)
app.add_page(index, title="EIDAP | Dashboard", on_load=AppState.load_data)
app.add_page(customer_360, route="/customer", title="EIDAP | Customer 360", on_load=AppState.load_data)
app.add_page(churn_intelligence, route="/churn", title="EIDAP | Churn", on_load=AppState.load_data)
app.add_page(clv_leaderboard, route="/clv", title="EIDAP | Lifetime Value", on_load=AppState.load_data)
app.add_page(forecasting, route="/forecast", title="EIDAP | Forecast", on_load=AppState.load_data)
app.add_page(assistant, route="/assistant", title="EIDAP | Assistant", on_load=AppState.load_data)
