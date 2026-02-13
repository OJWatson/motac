from motac.core.types import EventRecord, Dataset, Substrate

def test_core_types_construct():
    e = EventRecord(event_id='e0', t=0, lat=0.0, lon=0.0, cell_id=1, mark=None)
    assert e.event_id == 'e0'
    d = Dataset(events=None, counts=None, meta={})
    s = Substrate(grid=None, road_graph=None, neighbours=None, poi_features=None)
    assert d.meta == {}
    assert s.road_graph is None
