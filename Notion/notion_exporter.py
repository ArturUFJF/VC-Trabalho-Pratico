from notion_client import Client
from pathlib import Path
from dotenv import load_dotenv
import os



def send_to_notion(run_object, architecture, accuracy=0, loss=0, status="Finished"):
    directory = Path(__file__).resolve().parent.parent
    env = directory / '.env'
    load_dotenv(dotenv_path=env)

    notion_key = os.getenv("NOTION_KEY")
    notion_database = os.getenv("NOTION_DATABASE")

    if notion_key and notion_database:
        notion = Client(auth=notion_key.strip())
        print("✅ Notion conectado.")
    else:
        notion = None
        print("⚠️ Notion não configurado (verifique o .env).")

    if not notion:
        return
    try:
        notion.pages.create(
            parent={"database_id": notion_database.strip()},
            properties={
                "Runs": {
                    "title": [{"text": {"content": run_object.name}}]
                },

                "Architecture": {
                    "rich_text": [{"text": {"content": architecture}}]
                },

                "WandBLink": {
                    "url": run_object.url
                },
                "Accuracy": {
                    "number": float(accuracy)
                },
                "Loss": {
                    "number": float(loss)
                },
                "Status": {
                    "select": {"name": status}
                }
            }
        )
        print("✅ Dados salvos no Notion com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao salvar no Notion: {e}")