package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"html/template"
	"io"
	"log"
	"net/http"
	"os"
	"time"

	"ai-detector/internal/detector"
)

const (
	maxUploadSize = 5 << 20 // 5 MiB
	templatePath  = "templates/index.html"
	staticDir     = "static"
)

var (
	tmpl = template.Must(template.ParseFiles(templatePath))
)

type pageData struct {
	ErrorMessage string
	Result       *resultView
}

type resultView struct {
	Label      string
	Confidence string
	RawJSON    string
	PreviewSRC string
}

func main() {
	detectorClient, err := detector.NewFromEnv()
	if err != nil {
		log.Fatalf("detector configuration error: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", getHandler)
	mux.HandleFunc("/analyze", func(w http.ResponseWriter, r *http.Request) {
		analyzeHandler(w, r, detectorClient)
	})
	mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir(staticDir))))

	addr := fmt.Sprintf(":%s", getenv("PORT", "8080"))
	srv := &http.Server{
		Addr:         addr,
		Handler:      loggingMiddleware(mux),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	log.Printf("listening on %s", addr)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("server error: %v", err)
	}
}

func getHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Redirect(w, r, "/", http.StatusSeeOther)
		return
	}

	renderTemplate(w, pageData{})
}

func analyzeHandler(w http.ResponseWriter, r *http.Request, detectorClient *detector.HuggingFace) {
	if r.Method != http.MethodPost {
		http.Redirect(w, r, "/", http.StatusSeeOther)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxUploadSize)
	if err := r.ParseMultipartForm(maxUploadSize); err != nil {
		renderTemplate(w, pageData{ErrorMessage: "Upload failed. Please ensure the image is under 5 MB."})
		return
	}

	file, header, err := r.FormFile("image")
	if err != nil {
		renderTemplate(w, pageData{ErrorMessage: "Please choose an image file to upload."})
		return
	}
	defer file.Close()

	imageBytes, err := io.ReadAll(file)
	if err != nil || len(imageBytes) == 0 {
		renderTemplate(w, pageData{ErrorMessage: "Could not read uploaded file."})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 20*time.Second)
	defer cancel()

	result, err := detectorClient.Analyze(ctx, imageBytes)
	if err != nil {
		log.Printf("analyze error: %v", err)
		renderTemplate(w, pageData{ErrorMessage: "The detection service returned an error. Please try again later."})
		return
	}

	mimeType := header.Header.Get("Content-Type")
	if mimeType == "" {
		mimeType = "image/jpeg"
	}

	preview := base64.StdEncoding.EncodeToString(imageBytes)

	renderTemplate(w, pageData{
		Result: &resultView{
			Label:      result.Label,
			Confidence: result.ConfidencePercent(),
			RawJSON:    result.RawJSON,
			PreviewSRC: fmt.Sprintf("data:%s;base64,%s", mimeType, preview),
		},
	})
}

func renderTemplate(w http.ResponseWriter, data pageData) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := tmpl.Execute(w, data); err != nil {
		log.Printf("template execution error: %v", err)
	}
}

func getenv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}
