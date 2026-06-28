import { useQuery } from "@tanstack/react-query";
import { fetchSamples } from "../api/client";

export function useSamples() {
  return useQuery({
    queryKey: ["samples"],
    queryFn: fetchSamples,
  });
}
