# Knowledge Base

Drop documents here and they are ingested automatically the next time the backend starts.

Supported: `.pdf` `.docx` `.xlsx` `.xls` `.csv` `.txt`

```bash
cp ~/reports/q4-summary.pdf app/kb/
docker compose restart backend
```

Every file in this folder is indexed into the shared knowledge base under the `kb_default` session, so it stays queryable across all chat sessions — unlike files uploaded through the UI, which belong to the uploading session only.

Set `AUTO_INGEST_ON_STARTUP=false` to skip this on boot, or point `KB_FOLDER` elsewhere.

## This folder is gitignored

Its contents are excluded by `.gitignore` so personal or client documents are never committed. Only this README and `.gitkeep` are tracked.

Before committing any document deliberately, check what its metadata reveals — Office files record the author, organisation, and revision history:

```bash
unzip -p yourfile.docx docProps/core.xml
```
