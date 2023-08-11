from llama_index import download_loader, VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
from pathlib import Path
import os
import openai

os.environ["OPENAI_API_KEY"] = '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'
openai.api_key = os.environ["OPENAI_API_KEY"]

UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)
loader = UnstructuredReader()
file_path = Path('data/arquivo.txt')
docs = loader.load_data(file=file_path, split_documents=False)

service_context = ServiceContext.from_defaults(chunk_size=512)
storage_context = StorageContext.from_defaults()
index = VectorStoreIndex.from_documents(
  docs, 
  service_context=service_context,
  storage_context=storage_context,
)
storage_context.persist(persist_dir=f'./storage')
storage_context = StorageContext.from_defaults(persist_dir=f'./storage')
index = load_index_from_storage(storage_context=storage_context)

