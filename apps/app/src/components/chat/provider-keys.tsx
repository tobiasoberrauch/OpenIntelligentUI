"use client";

import { createContext, useContext, useState, type ReactNode } from "react";

export type ProviderKeys = { openai: string };
type ProviderKeysContextValue = {
  /** Visitors bring their own OpenAI key (NEXT_PUBLIC_BYOK=true). Off: the agent's server-side model is used. */
  byok: boolean;
  headers: Record<string, string> | undefined;
  hasKeys: boolean;
  session: number;
  save: (keys: ProviderKeys) => void;
  clear: () => void;
  newChat: () => void;
  draft: string;
  setDraft: (value: string) => void;
  keysOpen: boolean;
  setKeysOpen: (value: boolean) => void;
};
const ProviderKeysContext = createContext<ProviderKeysContextValue | null>(null);

// Read at render time: Next.js inlines NEXT_PUBLIC_* at build time; tests can stub it.
export function byokEnabled() {
  return process.env.NEXT_PUBLIC_BYOK === "true";
}

export function ProviderKeysProvider({ children }: { children: ReactNode }) {
  // Deliberately memory-only: never put credentials in browser storage or agent state.
  const [keys, setKeys] = useState<ProviderKeys | null>(null);
  const [session, setSession] = useState(0);
  const [draft, setDraft] = useState("");
  const [keysOpen, setKeysOpen] = useState(false);
  const byok = byokEnabled();
  const newChat = () => { setDraft(""); setSession((value) => value + 1); };
  return (
    <ProviderKeysContext.Provider value={{
      byok,
      headers: byok && keys ? { "x-openai-api-key": keys.openai } : undefined,
      hasKeys: !byok || keys !== null,
      session,
      save: (value) => { setKeys(value); setKeysOpen(false); setSession((value) => value + 1); },
      clear: () => { setKeys(null); newChat(); },
      newChat, draft, setDraft, keysOpen: byok && keysOpen, setKeysOpen,
    }}>
      {children}
    </ProviderKeysContext.Provider>
  );
}

export function useProviderKeys() {
  const value = useContext(ProviderKeysContext);
  if (!value) throw new Error("ProviderKeysProvider is required");
  return value;
}
