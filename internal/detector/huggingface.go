package detector

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

const defaultModelID = "orionw/ai-image-detection"

type HuggingFace struct {
	client *http.Client
	apiURL string
	token  string
}

type DetectionResult struct {
	Label   string
	Score   float64
	RawJSON string
}

func (r DetectionResult) ConfidencePercent() string {
	return fmt.Sprintf("%.1f%%", r.Score*100)
}

type apiResult struct {
	Label string  `json:"label"`
	Score float64 `json:"score"`
}

// NewFromEnv constructs a HuggingFace detector using environment variables.
// HF_API_URL overrides the automatic URL generation based on HF_MODEL_ID.
// HF_TOKEN is optional but recommended for higher rate limits.
func NewFromEnv() (*HuggingFace, error) {
	modelID := getenv("HF_MODEL_ID", defaultModelID)
	apiURL := os.Getenv("HF_API_URL")
	if apiURL == "" {
		apiURL = fmt.Sprintf("https://api-inference.huggingface.co/models/%s", modelID)
	}

	if apiURL == "" {
		return nil, errors.New("hugging face API URL is not configured")
	}

	token := os.Getenv("HF_TOKEN")

	return &HuggingFace{
		client: &http.Client{Timeout: 25 * time.Second},
		apiURL: apiURL,
		token:  token,
	}, nil
}

// Analyze sends the image bytes to the Hugging Face inference endpoint and
// returns the highest-confidence prediction.
func (h *HuggingFace) Analyze(ctx context.Context, image []byte) (DetectionResult, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.apiURL, bytes.NewReader(image))
	if err != nil {
		return DetectionResult{}, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Accept", "application/json")
	if h.token != "" {
		req.Header.Set("Authorization", "Bearer "+h.token)
	}

	resp, err := h.client.Do(req)
	if err != nil {
		return DetectionResult{}, fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return DetectionResult{}, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode == http.StatusServiceUnavailable {
		return DetectionResult{}, errors.New("model is loading on Hugging Face, please retry shortly")
	}

	if resp.StatusCode >= 400 {
		return DetectionResult{}, fmt.Errorf("hugging face API error %d: %s", resp.StatusCode, truncate(string(body), 200))
	}

	top, err := parseTopResult(body)
	if err != nil {
		return DetectionResult{}, err
	}

	indented, err := indentJSON(body)
	if err != nil {
		indented = string(body)
	}

	return DetectionResult{
		Label:   top.Label,
		Score:   top.Score,
		RawJSON: indented,
	}, nil
}

func parseTopResult(body []byte) (apiResult, error) {
	var list []apiResult
	if err := json.Unmarshal(body, &list); err == nil && len(list) > 0 {
		return list[0], nil
	}

	var nested [][]apiResult
	if err := json.Unmarshal(body, &nested); err == nil && len(nested) > 0 && len(nested[0]) > 0 {
		return nested[0][0], nil
	}

	var single apiResult
	if err := json.Unmarshal(body, &single); err == nil && single.Label != "" {
		return single, nil
	}

	return apiResult{}, errors.New("unexpected response format from Hugging Face API")
}

func indentJSON(raw []byte) (string, error) {
	var buf bytes.Buffer
	if err := json.Indent(&buf, raw, "", "  "); err != nil {
		return "", err
	}
	return buf.String(), nil
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

func getenv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
