"""HTTP contract: status codes, error shape, pagination and filters."""

from __future__ import annotations

import pytest


class TestHealth:
    def test_ok_when_data_is_loaded(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok" and body["trades_loaded"] > 0

    def test_degraded_rather_than_failing_when_empty(self, empty_client):
        """A health check must answer even when the service is useless, so an
        orchestrator can tell 'no data' apart from 'process is down'."""
        response = empty_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"


class TestTrades:
    def test_default_page(self, client):
        body = client.get("/trades").json()
        assert len(body["items"]) == 100 and body["total"] > 100 and body["has_more"]

    def test_offset_advances_the_window(self, client):
        first = client.get("/trades?limit=5").json()["items"]
        second = client.get("/trades?limit=5&offset=5").json()["items"]
        assert {t["trade_id"] for t in first}.isdisjoint({t["trade_id"] for t in second})

    def test_total_is_independent_of_the_page(self, client):
        assert client.get("/trades?limit=1").json()["total"] == client.get("/trades?limit=500").json()["total"]

    def test_last_page_reports_no_more(self, client):
        total = client.get("/trades?limit=1").json()["total"]
        assert client.get(f"/trades?limit=10&offset={total - 5}").json()["has_more"] is False

    def test_offset_past_the_end_is_empty_not_an_error(self, client):
        body = client.get("/trades?offset=99999999").json()
        assert body["items"] == [] and body["has_more"] is False

    @pytest.mark.parametrize("field,value", [
        ("symbol", "NVDA"), ("side", "BUY"), ("status", "REJECTED"),
    ])
    def test_filters_are_applied(self, client, field, value):
        key = "order_status" if field == "status" else field
        items = client.get(f"/trades?{field}={value}&limit=50").json()["items"]
        assert items and all(t[key] == value for t in items)

    def test_symbol_filter_is_case_insensitive(self, client):
        assert client.get("/trades?symbol=nvda&limit=1").json()["total"] == \
               client.get("/trades?symbol=NVDA&limit=1").json()["total"]

    def test_filters_compose_as_and(self, client):
        items = client.get("/trades?symbol=AAPL&side=SELL&limit=20").json()["items"]
        assert all(t["symbol"] == "AAPL" and t["side"] == "SELL" for t in items)

    def test_min_quantity(self, client):
        items = client.get("/trades?min_quantity=5000&limit=20").json()["items"]
        assert all(t["quantity"] >= 5000 for t in items)

    def test_unknown_symbol_returns_an_empty_page(self, client):
        assert client.get("/trades?symbol=ZZZZ").json()["total"] == 0

    def test_lookup_by_id(self, client):
        target = client.get("/trades?limit=1").json()["items"][0]["trade_id"]
        assert client.get(f"/trades/{target}").json()["trade_id"] == target

    def test_unknown_id_is_404_with_the_shared_error_shape(self, client):
        response = client.get("/trades/NOPE")
        assert response.status_code == 404
        assert response.json() == {
            "error": "trade_not_found",
            "detail": "No trade with id 'NOPE' in the current snapshot.",
        }


class TestValidationErrors:
    @pytest.mark.parametrize("query", [
        "limit=0", "limit=-1", "limit=100000", "offset=-1",
        "status=PENDING", "side=SIDEWAYS", "min_quantity=0",
    ])
    def test_bad_query_parameters_are_422(self, client, query):
        response = client.get(f"/trades?{query}")
        assert response.status_code == 422
        assert response.json()["error"] == "validation_error"

    def test_limit_cap_is_enforced_by_the_schema(self, client):
        """The cap has to live in the query constraint, not in handler code -
        otherwise the next route added forgets it."""
        assert client.get("/trades?limit=1001").status_code == 422
        assert client.get("/trades?limit=1000").status_code == 200


class TestMetrics:
    def test_global_metrics(self, client):
        body = client.get("/metrics").json()
        assert body["total_trades"] > 0
        assert 0 <= body["failure_rate"] <= 1
        assert body["avg_latency_ms"] <= body["p95_latency_ms"] <= body["p99_latency_ms"]

    def test_symbol_metrics(self, client):
        assert client.get("/metrics/symbol/AAPL").json()["symbol"] == "AAPL"

    def test_symbol_lookup_is_case_insensitive(self, client):
        assert client.get("/metrics/symbol/aapl").status_code == 200

    def test_unknown_symbol_is_404(self, client):
        response = client.get("/metrics/symbol/ZZZZ")
        assert response.status_code == 404 and response.json()["error"] == "symbol_not_found"

    def test_top_k_returns_k_descending(self, client):
        items = client.get("/metrics/top?sort_by=notional&k=5").json()
        values = [m["total_notional"] for m in items]
        assert len(items) == 5 and values == sorted(values, reverse=True)

    @pytest.mark.parametrize("sort_by", ["trade_count", "notional", "quantity", "failure_rate", "latency"])
    def test_every_documented_sort_key_works(self, client, sort_by):
        assert client.get(f"/metrics/top?sort_by={sort_by}&k=3").status_code == 200

    def test_unknown_sort_key_is_rejected(self, client):
        """`sort_by` indexes an allow-list, never `getattr` on user input."""
        assert client.get("/metrics/top?sort_by=__class__").status_code == 404


class TestAnomalies:
    def test_listing(self, client):
        body = client.get("/anomalies").json()
        assert body["total"] > 0

    def test_severity_filter(self, client):
        items = client.get("/anomalies?severity=CRITICAL&limit=10").json()["items"]
        assert all(a["severity"] == "CRITICAL" for a in items)

    def test_type_filter(self, client):
        items = client.get("/anomalies?anomaly_type=HIGH_LATENCY&limit=10").json()["items"]
        assert items and all(a["anomaly_type"] == "HIGH_LATENCY" for a in items)

    def test_every_anomaly_carries_its_explanation(self, client):
        for anomaly in client.get("/anomalies?limit=50").json()["items"]:
            assert anomaly["reason"] and anomaly["metric_name"]
            assert anomaly["metric_value"] is not None and anomaly["threshold"] is not None

    def test_unknown_type_returns_empty(self, client):
        assert client.get("/anomalies?anomaly_type=NOT_A_RULE").json()["total"] == 0


class TestSignals:
    def test_listing(self, client):
        assert client.get("/signals").json()["total"] > 0

    def test_action_filter(self, client):
        for signal in client.get("/signals?action=HOLD").json()["items"]:
            assert signal["action"] == "HOLD"

    def test_confidence_filter(self, client):
        for signal in client.get("/signals?min_confidence=0.5").json()["items"]:
            assert signal["confidence"] >= 0.5

    def test_invalid_confidence_is_422(self, client):
        assert client.get("/signals?min_confidence=1.5").status_code == 422


class TestEmptyDataset:
    @pytest.mark.parametrize("path", ["/trades", "/signals", "/anomalies", "/metrics", "/metrics/symbol/AAPL"])
    def test_data_endpoints_return_503_not_500(self, empty_client, path):
        """'No data loaded' is an operator problem, not a bug. 503 says so; a
        500 would send someone hunting for a crash that never happened."""
        response = empty_client.get(path)
        assert response.status_code == 503
        assert response.json()["error"] == "dataset_empty"


class TestOpenApi:
    def test_schema_is_generated(self, client):
        schema = client.get("/openapi.json").json()
        for path in ("/health", "/trades", "/trades/{trade_id}", "/signals", "/anomalies", "/metrics"):
            assert path in schema["paths"]

    def test_docs_render(self, client):
        assert client.get("/docs").status_code == 200
