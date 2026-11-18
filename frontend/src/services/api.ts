import type { HealthResponse } from "../types/api";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ??
  "http://localhost:8000/api/v1"
).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function apiRequest<T>(endpoint: string): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    });
  } catch {
    throw new ApiError(
      "Unable to connect to the AURA backend.",
      0
    );
  }

  if (!response.ok) {
    throw new ApiError(
      `Backend request failed with status ${response.status}.`,
      response.status
    );
  }

  return response.json() as Promise<T>;
}

export function getHealth(): Promise<HealthResponse> {
  return apiRequest<HealthResponse>("/health");
}
