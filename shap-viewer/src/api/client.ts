import mockData from "./mock-data.json";
import type { SamplesResponse } from "../types/shap";

const USE_MOCK = import.meta.env.VITE_USE_MOCK_DATA === "true";
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

export async function fetchSamples(): Promise<SamplesResponse> {
  if (USE_MOCK) {
    await new Promise((resolve) => setTimeout(resolve, 150));
    return mockData as SamplesResponse;
  }

  const response = await fetch(`${API_BASE_URL}/samples`, {
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch samples: ${response.status}`);
  }

  return response.json();
}
