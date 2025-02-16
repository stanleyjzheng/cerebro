"use client";

import React, { useState, useEffect } from "react";
import ImageUpload from "@/components/ImageUpload"; // Adjust the path based on your project structure.

const API_BASE = "http://localhost:8000";
const DEFAULT_TOP_N = 5;

export default function SearchPage() {
  const [searchType, setSearchType] = useState<"text" | "image">("text");
  const [filter, setFilter] = useState<null | "image" | "youtube">(null);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Debounce the text query.
  useEffect(() => {
    if (searchType !== "text") return;
    if (query.trim() === "") {
      setResults([]);
      return;
    }
    const timer = setTimeout(() => {
      searchText(query);
    }, 500);
    return () => clearTimeout(timer);
  }, [query, filter, searchType]);

  // API call for text search.
  const searchText = async (q: string) => {
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams({
        query: q,
        top_n: DEFAULT_TOP_N.toString(),
      });
      if (filter) params.append("filter_media", filter);
      const res = await fetch(`${API_BASE}/search/text?${params.toString()}`);
      if (!res.ok) throw new Error("Error fetching text results");
      const data = await res.json();
      setResults(data);
    } catch (e: any) {
      setError(e.message || "Unknown error occurred");
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  // API call for image search.
  const searchImage = async (selectedFile: File) => {
    setLoading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("top_n", DEFAULT_TOP_N.toString());
      if (filter) formData.append("filter_media", filter);
      const res = await fetch(`${API_BASE}/search/image`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Error fetching image results");
      const data = await res.json();
      setResults(data);
    } catch (e: any) {
      setError(e.message || "Unknown error occurred");
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  // Update the filter state.
  const handleFilterChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    if (val === "All") setFilter(null);
    else if (val === "Images") setFilter("image");
    else if (val === "Videos") setFilter("youtube");
  };

  // Switch between search types.
  const switchSearchType = (type: "text" | "image") => {
    setSearchType(type);
    setResults([]);
    setQuery("");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-blue-50 font-sans">
      <div className="container mx-auto px-4 py-12">
        {/* Header */}
        <header className="mb-12 text-center">
          <h1 className="text-5xl font-extrabold text-gray-900 mb-4">Cerebro</h1>
          <p className="text-lg text-gray-600">
            Instantly search 150k+ wildlife images with semantics search. Or, upload a drawing to see it in real life.
          </p>
        </header>

        {/* Search Card */}
        <div className="max-w-3xl mx-auto bg-white/80 backdrop-blur-md shadow-xl rounded-2xl overflow-hidden">
          {/* Tabs */}
          <div className="flex border-b">
            <button
              onClick={() => switchSearchType("text")}
              className={`flex-1 py-4 text-center transition-colors duration-200 font-semibold ${
                searchType === "text" ? "bg-blue-600 text-white" : "bg-white text-gray-700 hover:bg-gray-100"
              }`}
            >
              Text
            </button>
            <button
              onClick={() => switchSearchType("image")}
              className={`flex-1 py-4 text-center transition-colors duration-200 font-semibold ${
                searchType === "image" ? "bg-blue-600 text-white" : "bg-white text-gray-700 hover:bg-gray-100"
              }`}
            >
              Image
            </button>
          </div>

          {/* Form */}
          <div className="p-8">
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">Media Filter</label>
              <select
                onChange={handleFilterChange}
                className="w-full border border-gray-300 rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option>All</option>
                <option>Images</option>
                <option>Videos</option>
              </select>
            </div>

            {searchType === "text" && (
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">Enter your query</label>
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search for something..."
                  className="w-full border border-gray-300 rounded-md px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-800"
                />
              </div>
            )}

            {searchType === "image" && (
              <div className="mb-6">
                <ImageUpload onFileChange={searchImage} />
              </div>
            )}

            {loading && <p className="text-center text-gray-500">Searching...</p>}
            {error && <p className="text-center text-red-500">{error}</p>}
          </div>
        </div>

        {/* Results */}
        {results.length > 0 && (
          <section className="mt-12">
            <h2 className="text-3xl font-bold text-gray-900 text-center mb-8">Results</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-8">
              {results.map((item, idx) => (
                <div
                  key={idx}
                  className="bg-white rounded-2xl shadow-md overflow-hidden transform transition hover:scale-105"
                >
                  {item.media_type === "image" && item.file_path && (
                    <img
                      src={item.file_path.replace("../dataset/", "/dataset/")}
                      alt="Result"
                      className="w-full h-56 object-cover"
                    />
                  )}
                  <div className="p-4">
                    {item.youtube_video_id && (
                      <a
                        href={`https://www.youtube.com/watch?v=${item.youtube_video_id}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block text-blue-600 hover:underline font-medium mb-2"
                      >
                        Watch on YouTube
                      </a>
                    )}
                    <div className="text-sm text-gray-600">
                      <p>Score: {item.score?.toFixed(2)}</p>
                      {item.timestamp && <p>Timestamp: {item.timestamp}</p>}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
