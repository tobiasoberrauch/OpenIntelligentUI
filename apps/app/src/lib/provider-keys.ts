export type ProviderHeaders = {
  "x-openai-api-key": string;
};

export const PROVIDER_KEY_INPUT_ERROR = "Enter an OpenAI API key (up to 4,096 characters, without spaces).";

// Credentials travel only as explicitly selected headers, never as agent state.
export function readProviderHeaders(headers: Headers): ProviderHeaders | undefined {
  const openai = headers.get("x-openai-api-key");
  if (openai === null) return undefined;
  if (!/^[\x21-\x7e]{1,4096}$/.test(openai.trim())) throw new Error(PROVIDER_KEY_INPUT_ERROR);
  return { "x-openai-api-key": openai.trim() };
}
