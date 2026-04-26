import reflex as rx

config = rx.Config(
    app_name="edi_app",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
)