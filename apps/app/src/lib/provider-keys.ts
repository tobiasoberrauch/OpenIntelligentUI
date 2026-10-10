export type ProviderHeaders = {
  "x-openai-api-key": string;
  /** Optional: selects Jev visualization routing for this visitor. */
  "x-jev-api-key"?: string;
};

export const PROVIDER_KEY_INPUT_ERROR = "Enter an OpenAI API key and, optionally, a Jev API key (up to 4,096 characters each, without spaces).";

const valid = (value: string) => /^[\x21-\x7e]{1,4096}$/.test(value.trim());

// Credentials travel only as explicitly selected headers, never as agent state.
export function readProviderHeaders(headers: Headers): ProviderHeaders | undefined {
  const openai = headers.get("x-openai-api-key");
  const jev = headers.get("x-jev-api-key");
  if (openai === null && jev === null) return undefined;
  // A Jev key alone is invalid: answers always need the visitor's OpenAI key.
  if (openai === null || !valid(openai)) throw new Error(PROVIDER_KEY_INPUT_ERROR);
  if (jev !== null && !valid(jev)) throw new Error(PROVIDER_KEY_INPUT_ERROR);
  return jev === null
    ? { "x-openai-api-key": openai.trim() }
    : { "x-openai-api-key": openai.trim(), "x-jev-api-key": jev.trim() };
}
