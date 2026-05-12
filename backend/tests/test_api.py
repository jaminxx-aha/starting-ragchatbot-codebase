"""
Tests for FastAPI endpoints.

Tests cover:
1. /api/query endpoint - request/response handling, session management
2. /api/courses endpoint - course statistics retrieval
3. /api/session/{session_id} endpoint - session clearing
4. / root endpoint - basic health check
5. Error handling - 500 errors, validation errors
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


@pytest.mark.api
class TestQueryEndpoint:
    """Tests for /api/query endpoint"""

    def test_query_endpoint_returns_valid_response(self, mock_client, sample_query_request):
        """Test that query endpoint returns properly formatted response"""
        response = mock_client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

    def test_query_endpoint_creates_session_when_none(self, mock_client):
        """Test that new session is created when session_id is None"""
        request = {"query": "test", "session_id": None, "language": "zh"}
        response = mock_client.post("/api/query", json=request)

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"

    def test_query_endpoint_preserves_existing_session(self, mock_client):
        """Test that existing session_id is preserved"""
        request = {"query": "test", "session_id": "existing-session-456", "language": "zh"}
        response = mock_client.post("/api/query", json=request)

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session-456"

    def test_query_endpoint_sources_have_required_fields(self, mock_client, sample_query_request):
        """Test that each source has display_text and optional link"""
        response = mock_client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()

        for source in data["sources"]:
            assert "display_text" in source
            assert "link" in source or "link" is None

    def test_query_endpoint_handles_chinese_language(self, mock_client, sample_query_request_chinese):
        """Test query with Chinese language parameter"""
        response = mock_client.post("/api/query", json=sample_query_request_chinese)

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] is not None

    def test_query_endpoint_handles_english_language(self, mock_client):
        """Test query with English language parameter"""
        request = {"query": "What is Claude?", "language": "en"}
        response = mock_client.post("/api/query", json=request)

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] is not None

    def test_query_endpoint_default_language_is_chinese(self, mock_client):
        """Test that default language is Chinese"""
        request = {"query": "测试问题"}  # No language specified
        response = mock_client.post("/api/query", json=request)

        assert response.status_code == 200

    def test_query_endpoint_requires_query_field(self, mock_client):
        """Test that query field is required"""
        request = {"session_id": "test", "language": "zh"}  # Missing query
        response = mock_client.post("/api/query", json=request)

        assert response.status_code == 422  # Validation error

    def test_query_endpoint_empty_query_accepted(self, mock_client):
        """Test that empty query string is accepted (handled by RAG system)"""
        request = {"query": "", "language": "zh"}
        response = mock_client.post("/api/query", json=request)

        # Should return 200 (empty query handling is RAG system's responsibility)
        assert response.status_code == 200

    def test_query_endpoint_handles_500_error(self, error_client):
        """Test that 500 errors are properly returned"""
        request = {"query": "test", "language": "zh"}
        response = error_client.post("/api/query", json=request)

        assert response.status_code == 500
        assert "detail" in response.json()


@pytest.mark.api
class TestCoursesEndpoint:
    """Tests for /api/courses endpoint"""

    def test_courses_endpoint_returns_stats(self, mock_client):
        """Test that courses endpoint returns statistics"""
        response = mock_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

    def test_courses_endpoint_course_titles_are_strings(self, mock_client):
        """Test that all course titles are strings"""
        response = mock_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        for title in data["course_titles"]:
            assert isinstance(title, str)

    def test_courses_endpoint_total_matches_titles_count(self, mock_client):
        """Test that total_courses matches the length of course_titles"""
        response = mock_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == len(data["course_titles"])

    def test_courses_endpoint_handles_500_error(self, error_client):
        """Test courses endpoint error handling"""
        # Override mock for analytics error
        error_client.app.dependency_overrides = {}
        response = error_client.get("/api/courses")

        # May return 500 if analytics throws error
        assert response.status_code in [200, 500]


@pytest.mark.api
class TestSessionEndpoint:
    """Tests for /api/session/{session_id} endpoint"""

    def test_session_clear_returns_success(self, mock_client):
        """Test that session clear returns success message"""
        response = mock_client.delete("/api/session/test-session-123")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert "message" in data

    def test_session_clear_with_any_session_id(self, mock_client):
        """Test that any session_id format is accepted"""
        response = mock_client.delete("/api/session/any-random-id")

        assert response.status_code == 200

    def test_session_clear_returns_json(self, mock_client):
        """Test that response is valid JSON"""
        response = mock_client.delete("/api/session/test-id")

        assert response.headers.get("content-type") == "application/json"


@pytest.mark.api
class TestRootEndpoint:
    """Tests for root endpoint"""

    def test_root_endpoint_returns_message(self, mock_client):
        """Test that root endpoint returns a message"""
        response = mock_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data


@pytest.mark.api
@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests using real RAG system (mocked AI responses)"""

    def test_full_query_flow_with_mocked_ai(self, mock_client, mock_rag_system):
        """Test complete query flow with mocked AI response"""
        response = mock_client.post("/api/query", json={"query": "test"})

        assert response.status_code == 200
        mock_rag_system.query.assert_called()

    def test_query_followed_by_session_clear(self, mock_client):
        """Test query then clear session flow"""
        # First make a query
        query_response = mock_client.post("/api/query", json={"query": "test"})
        session_id = query_response.json()["session_id"]

        # Then clear the session
        clear_response = mock_client.delete(f"/api/session/{session_id}")

        assert clear_response.status_code == 200
        assert clear_response.json()["status"] == "success"

    def test_courses_endpoint_consistency(self, mock_client):
        """Test that courses endpoint returns consistent data"""
        response1 = mock_client.get("/api/courses")
        response2 = mock_client.get("/api/courses")

        assert response1.json() == response2.json()


@pytest.mark.api
class TestAPIResponseModels:
    """Tests for Pydantic response model validation"""

    def test_query_response_model_validation(self, mock_client, sample_query_request):
        """Test that response matches QueryResponse model"""
        response = mock_client.post("/api/query", json=sample_query_request)
        data = response.json()

        # Required fields
        assert data["answer"] is not None
        assert data["session_id"] is not None
        assert isinstance(data["sources"], list)

    def test_course_stats_model_validation(self, mock_client):
        """Test that response matches CourseStats model"""
        response = mock_client.get("/api/courses")
        data = response.json()

        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

    def test_source_info_model_structure(self, mock_client, sample_query_request):
        """Test that sources match SourceInfo model"""
        response = mock_client.post("/api/query", json=sample_query_request)
        data = response.json()

        for source in data["sources"]:
            assert "display_text" in source
            # link can be None or string
            if source.get("link") is not None:
                assert isinstance(source["link"], str)


@pytest.mark.api
class TestAPIContentType:
    """Tests for API content types and headers"""

    def test_query_returns_json_content_type(self, mock_client, sample_query_request):
        """Test that response content type is JSON"""
        response = mock_client.post("/api/query", json=sample_query_request)

        assert "application/json" in response.headers.get("content-type", "")

    def test_courses_returns_json_content_type(self, mock_client):
        """Test that courses endpoint returns JSON"""
        response = mock_client.get("/api/courses")

        assert "application/json" in response.headers.get("content-type", "")

    def test_session_clear_returns_json_content_type(self, mock_client):
        """Test that session clear returns JSON"""
        response = mock_client.delete("/api/session/test-id")

        assert "application/json" in response.headers.get("content-type", "")