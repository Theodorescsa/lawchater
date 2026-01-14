from core.app import get_rag_service
rag = get_rag_service()
print("\n--- DANH SÁCH FILE TRONG DB ---")
data = rag.vectorstore.get()
all_sources = set()
for meta in data['metadatas']:
    if meta and 'source_name' in meta:
        all_sources.add(meta['source_name'])
for s in all_sources:
    print(f"'{s}'")
print("-------------------------------")