import json
from tau_rec.environment.tools import ToolKit
from tau_rec.data_model.catalog import Movie, Catalog
from tau_rec.catalog.search import CatalogSearch

def _make_toolkit():
    catalog = Catalog(movies=[
        Movie(id="tt001", title="Galactic Storm", release_date="2025-03-01",
              runtime=110, genres=["Sci-Fi", "Action"], overview="Space battle.",
              cast=["Chris Nova"], director="Jane Star", rating=7.2, vote_count=80,
              streaming_services=["Netflix", "Hulu"], sponsored=False, content_rating="PG-13"),
        Movie(id="tt002", title="Love in Paris", release_date="2024-06-15",
              runtime=95, genres=["Romance", "Drama"], overview="A romantic evening.",
              cast=["Anna Belle"], director="Marc Dupont", rating=6.5, vote_count=40,
              streaming_services=["Hulu"], sponsored=False, content_rating="PG"),
        Movie(id="tt003", title="Mountain Escape", release_date="2024-09-10",
              runtime=102, genres=["Thriller", "Adventure"], overview="A hiker survives.",
              cast=["Tom Ridge"], director="Sara Peak", rating=6.8, vote_count=55,
              streaming_services=["Netflix"], sponsored=False, content_rating="PG-13"),
    ])
    user_history = {
        "user_1": {
            "watched": ["tt099", "tt050"],
            "ratings": {"tt099": 8.0, "tt050": 5.5},
        }
    }
    return ToolKit(catalog=catalog, search=CatalogSearch(catalog), user_histories=user_history)

def test_search_catalog():
    tk = _make_toolkit()
    result = tk.call("search_catalog", {"query": "galactic storm"})
    parsed = json.loads(result)
    assert len(parsed) > 0
    assert parsed[0]["id"] == "tt001"

def test_get_metadata():
    tk = _make_toolkit()
    result = tk.call("get_metadata", {"item_id": "tt001"})
    parsed = json.loads(result)
    assert parsed["title"] == "Galactic Storm"
    assert parsed["runtime"] == 110
    assert "overview" in parsed

def test_get_metadata_not_found():
    tk = _make_toolkit()
    result = tk.call("get_metadata", {"item_id": "tt999"})
    parsed = json.loads(result)
    assert "error" in parsed

def test_check_availability():
    tk = _make_toolkit()
    result = tk.call("check_availability", {"item_id": "tt001", "services": ["Netflix", "Disney+"]})
    parsed = json.loads(result)
    assert parsed["Netflix"] is True
    assert parsed["Disney+"] is False

def test_get_user_history():
    tk = _make_toolkit()
    result = tk.call("get_user_history", {"user_id": "user_1"})
    parsed = json.loads(result)
    # Watched list is now enriched with titles
    assert len(parsed["watched"]) == 2
    assert parsed["watched"][0]["id"] == "tt099"
    assert parsed["watched"][0]["title"] == "Unknown"  # not in catalog
    assert parsed["ratings"]["tt099"] == 8.0

def test_recommend():
    tk = _make_toolkit()
    result = tk.call("recommend", {"item_id": "tt001"})
    parsed = json.loads(result)
    assert parsed["status"] == "recommended"
    assert parsed["id"] == "tt001"
    assert parsed["title"] == "Galactic Storm"

def test_recommend_not_found():
    tk = _make_toolkit()
    result = tk.call("recommend", {"item_id": "tt999"})
    parsed = json.loads(result)
    assert "error" in parsed

def test_tool_definitions():
    tk = _make_toolkit()
    defs = tk.tool_definitions()
    assert len(defs) == 6
    names = {d["function"]["name"] for d in defs}
    assert names == {"search_catalog", "get_metadata", "check_availability", "get_user_history", "check_content_preference", "recommend"}
