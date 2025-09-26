from dotenv import load_dotenv
from typing import Dict, Any, Optional
from bson import ObjectId
import pymongo
import os
from pydantic import BaseModel, Field

from simple_or_agent.instructor_based.tools import ToolSpec

# --- Init ---

load_dotenv()
mongo_client: pymongo.MongoClient = pymongo.MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))

# --- Args Models ---

class MongoInsertOneArgs(BaseModel):
    """Inputs for the MongoDB insert_one tool."""
    db_name: str = Field(..., description="The name of the database.")
    collection_name: str = Field(..., description="The name of the collection.")
    document: Dict[str, Any] = Field(..., description="The JSON document to insert.")

class MongoFindOneArgs(BaseModel):
    """Inputs for the MongoDB find_one tool."""
    db_name: str = Field(..., description="The name of the database.")
    collection_name: str = Field(..., description="The name of the collection.")
    filter: Optional[Dict[str, Any]] = Field(None, description="A JSON object to filter the query.")

class MongoUpdateOneArgs(BaseModel):
    """Inputs for the MongoDB update_one tool."""
    db_name: str = Field(..., description="The name of the database.")
    collection_name: str = Field(..., description="The name of the collection.")
    filter: Dict[str, Any] = Field(..., description="A JSON object to select the document to update.")
    update: Dict[str, Any] = Field(..., description="A JSON object specifying the update operations (e.g., using $set).")

class MongoDeleteOneArgs(BaseModel):
    """Inputs for the MongoDB delete_one tool."""
    db_name: str = Field(..., description="The name of the database.")
    collection_name: str = Field(..., description="The name of the collection.")
    filter: Dict[str, Any] = Field(..., description="A JSON object to select the document to delete.")

class MongoFindArgs(BaseModel):
    """Inputs for the MongoDB find tool."""
    db_name: str = Field(..., description="The name of the database.")
    collection_name: str = Field(..., description="The name of the collection.")
    filter: Optional[Dict[str, Any]] = Field(None, description="A JSON object to filter the query.")
    limit: int = Field(25, description="The maximum number of documents to return.")

class MongoAggregateArgs(BaseModel):
    """Inputs for the MongoDB aggregate tool."""
    db_name: str = Field(..., description="The name of the database.")
    collection_name: str = Field(..., description="The name of the collection.")
    pipeline: list = Field(..., description="A list of JSON objects representing the aggregation stages.")

class MongoCountDocumentsArgs(BaseModel):
    """Inputs for the MongoDB count_documents tool."""
    db_name: str = Field(..., description="The name of the database.")
    collection_name: str = Field(..., description="The name of the collection.")
    filter: Optional[Dict[str, Any]] = Field(None, description="A JSON object to filter the documents to be counted.")

class MongoListCollectionsArgs(BaseModel):
    """Inputs for the MongoDB list_collections tool."""
    db_name: str = Field(..., description="The name of the database to inspect.")

class MongoDropCollectionArgs(BaseModel):
    """Inputs for the MongoDB drop_collection tool."""
    db_name: str = Field(..., description="The name of the database.")
    collection_name: str = Field(..., description="The name of the collection to drop.")

class MongoListDatabasesResponse(BaseModel):
    """Response from the MongoDB list_databases tool."""
    databases: list = Field(..., description="List of database names")

def _serialize_doc(doc: Any) -> Any:
    """Recursively convert ObjectId to string for JSON serialization."""
    if isinstance(doc, ObjectId):
        return str(doc)
    if isinstance(doc, list):
        return [_serialize_doc(item) for item in doc]
    if isinstance(doc, dict):
        return {key: _serialize_doc(value) for key, value in doc.items()}
    return doc

# --- CRUD Tools ---

def make_mongo_insert_one_tool() -> ToolSpec:
    """Inserts a single document into a collection."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            parsed_args = MongoInsertOneArgs(**args)
            db = mongo_client[parsed_args.db_name]
            collection = db[parsed_args.collection_name]
            result = collection.insert_one(parsed_args.document)
            return {"inserted_id": str(result.inserted_id)}
        except Exception as e:
            return {"error": f"mongo_insert_one failed: {e}"}

    return ToolSpec(
        name="mongo_insert_one",
        description="Inserts a single document into a specified MongoDB collection.",
        args_model=MongoInsertOneArgs,
        handler=handler,
        parameters={
            "db_name": "the name of the database",
            "collection_name": "the name of the collection",
            "document": "the JSON document to insert",
        },
    )

def make_mongo_find_one_tool() -> ToolSpec:
    """Finds a single document in a collection."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            parsed_args = MongoFindOneArgs(**args)
            db = mongo_client[parsed_args.db_name]
            collection = db[parsed_args.collection_name]
            doc = collection.find_one(parsed_args.filter or {})
            return _serialize_doc(doc) if doc else None
        except Exception as e:
            return {"error": f"mongo_find_one failed: {e}"}

    return ToolSpec(
        name="mongo_find_one",
        description="Finds a single document in a collection that matches the filter.",
        args_model=MongoFindOneArgs,
        handler=handler,
        parameters={
            "db_name": "the name of the database",
            "collection_name": "the name of the collection",
            "filter": "a JSON object to filter the query",
        },
    )

def make_mongo_update_one_tool() -> ToolSpec:
    """Updates a single document in a collection."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            parsed_args = MongoUpdateOneArgs(**args)
            db = mongo_client[parsed_args.db_name]
            collection = db[parsed_args.collection_name]
            result = collection.update_one(parsed_args.filter, parsed_args.update)
            return {
                "matched_count": result.matched_count,
                "modified_count": result.modified_count,
            }
        except Exception as e:
            return {"error": f"mongo_update_one failed: {e}"}

    return ToolSpec(
        name="mongo_update_one",
        description="Updates a single document matching the filter in a collection.",
        args_model=MongoUpdateOneArgs,
        handler=handler,
        parameters={
            "db_name": "the name of the database",
            "collection_name": "the name of the collection",
            "filter": "a JSON object to select the document to update",
            "update": "a JSON object specifying the update operations (e.g., using $set)",
        },
    )
    
def make_mongo_delete_one_tool() -> ToolSpec:
    """Deletes a single document from a collection."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            parsed_args = MongoDeleteOneArgs(**args)
            db = mongo_client[parsed_args.db_name]
            collection = db[parsed_args.collection_name]
            result = collection.delete_one(parsed_args.filter)
            return {"deleted_count": result.deleted_count}
        except Exception as e:
            return {"error": f"mongo_delete_one failed: {e}"}

    return ToolSpec(
        name="mongo_delete_one",
        description="Deletes a single document matching the filter from a collection.",
        args_model=MongoDeleteOneArgs,
        handler=handler,
        parameters={
            "db_name": "the name of the database",
            "collection_name": "the name of the collection",
            "filter": "a JSON object to select the document to delete",
        },
    )

# --- Advanced Data Tools ---

def make_mongo_find_tool() -> ToolSpec:
    """Finds multiple documents in a collection."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            parsed_args = MongoFindArgs(**args)
            db = mongo_client[parsed_args.db_name]
            collection = db[parsed_args.collection_name]
            cursor = collection.find(parsed_args.filter or {}).limit(parsed_args.limit)
            return [_serialize_doc(doc) for doc in cursor]
        except Exception as e:
            return {"error": f"mongo_find failed: {e}"}

    return ToolSpec(
        name="mongo_find",
        description="Finds multiple documents in a collection that match the filter. Returns up to 25 documents by default.",
        args_model=MongoFindArgs,
        handler=handler,
        parameters={
            "db_name": "the name of the database",
            "collection_name": "the name of the collection",
            "filter": "a JSON object to filter the query",
            "limit": "the maximum number of documents to return",
        },
    )

def make_mongo_aggregate_tool() -> ToolSpec:
    """Runs a MongoDB aggregation pipeline."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            parsed_args = MongoAggregateArgs(**args)
            db = mongo_client[parsed_args.db_name]
            collection = db[parsed_args.collection_name]
            cursor = collection.aggregate(parsed_args.pipeline)
            return [_serialize_doc(doc) for doc in cursor]
        except Exception as e:
            return {"error": f"mongo_aggregate failed: {e}"}

    return ToolSpec(
        name="mongo_aggregate",
        description="Performs complex data aggregation using a pipeline of stages.",
        args_model=MongoAggregateArgs,
        handler=handler,
        parameters={
            "db_name": "the name of the database",
            "collection_name": "the name of the collection",
            "pipeline": "a list of JSON objects representing the aggregation stages",
        },
    )

def make_mongo_count_documents_tool() -> ToolSpec:
    """Counts documents matching a filter."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            parsed_args = MongoCountDocumentsArgs(**args)
            db = mongo_client[parsed_args.db_name]
            collection = db[parsed_args.collection_name]
            count = collection.count_documents(parsed_args.filter or {})
            return {"count": count}
        except Exception as e:
            return {"error": f"mongo_count_documents failed: {e}"}

    return ToolSpec(
        name="mongo_count_documents",
        description="Counts the number of documents in a collection that match the given filter.",
        args_model=MongoCountDocumentsArgs,
        handler=handler,
        parameters={
            "db_name": "the name of the database",
            "collection_name": "the name of the collection",
            "filter": "a JSON object to filter the documents to be counted",
        },
    )

# --- Database and Collection Management Tools ---

def make_mongo_list_databases_tool() -> ToolSpec:
    """Lists all database names on the server."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            return {"databases": mongo_client.list_database_names()}
        except Exception as e:
            return {"error": f"mongo_list_databases failed: {e}"}

    return ToolSpec(
        name="mongo_list_databases",
        description="Lists the names of all databases on the MongoDB server.",
        response_model=MongoListDatabasesResponse,
        handler=handler,
        parameters={},
    )

def make_mongo_list_collections_tool() -> ToolSpec:
    """Lists all collection names in a database."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            parsed_args = MongoListCollectionsArgs(**args)
            db = mongo_client[parsed_args.db_name]
            return {"collections": db.list_collection_names()}
        except Exception as e:
            return {"error": f"mongo_list_collections failed: {e}"}

    return ToolSpec(
        name="mongo_list_collections",
        description="Lists the names of all collections within a specified database.",
        args_model=MongoListCollectionsArgs,
        handler=handler,
        parameters={
            "db_name": "the name of the database to inspect",
        },
    )

def make_mongo_drop_collection_tool() -> ToolSpec:
    """Drops (deletes) an entire collection."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            parsed_args = MongoDropCollectionArgs(**args)
            db = mongo_client[parsed_args.db_name]
            db.drop_collection(parsed_args.collection_name)
            return {"message": f"Collection '{parsed_args.collection_name}' dropped successfully from database '{parsed_args.db_name}'."}
        except Exception as e:
            return {"error": f"mongo_drop_collection failed: {e}"}

    return ToolSpec(
        name="mongo_drop_collection",
        description="Deletes an entire collection from a database. This action is irreversible.",
        args_model=MongoDropCollectionArgs,
        handler=handler,
        parameters={
            "db_name": "the name of the database",
            "collection_name": "the name of the collection to drop",
        },
    )

__all__ = [
    "make_mongo_insert_one_tool",
    "make_mongo_find_one_tool",
    "make_mongo_update_one_tool",
    "make_mongo_delete_one_tool",
    "make_mongo_find_tool",
    "make_mongo_aggregate_tool",
    "make_mongo_count_documents_tool",
    "make_mongo_list_databases_tool",
    "make_mongo_list_collections_tool",
    "make_mongo_drop_collection_tool",
]