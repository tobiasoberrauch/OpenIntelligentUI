"use client";

import { useEffect, useRef, useState } from "react";
import { useProviderKeys } from "./provider-keys";
import "./provider-keys.css";

export function ProviderKeysDialog({ onClose, disabled = false }: { onClose: () => void; disabled?: boolean }) {
  const { hasKeys, save, clear } = useProviderKeys();
  const dialog = useRef<HTMLDialogElement>(null);
  const request = useRef<AbortController | null>(null);
  const [openai, setOpenai] = useState("");
  const [pending, setPending] = useState(false);
  const [validated, setValidated] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    dialog.current?.showModal();
    return () => {
      request.current?.abort();
      opener?.focus();
    };
  }, []);

  const edit = (setter: (value: string) => void, value: string) => {
    setter(value);
    setValidated(false);
    setError(null);
  };
  const testConnection = async () => {
    if (pending || disabled || !openai.trim()) return;
    const controller = new AbortController();
    request.current = controller;
    setPending(true);
    setValidated(false);
    setError(null);
    try {
      const response = await fetch("/api/provider-keys", {
        method: "POST",
        headers: { "x-openai-api-key": openai.trim() },
        signal: controller.signal,
        cache: "no-store",
      });
      const result = await response.json();
      if (controller.signal.aborted) return;
      if (!response.ok || result.ok !== true) {
        // The same-origin proxy maps provider failures to fixed, sanitized messages.
        setError(typeof result.error === "string" ? result.error : "Could not validate the key. Check the key and provider access, then try again.");
      } else {
        setValidated(true);
      }
    } catch {
      if (!controller.signal.aborted) setError("Could not connect. Check your connection and try again.");
    } finally {
      if (!controller.signal.aborted) setPending(false);
    }
  };

  return (
    <dialog ref={dialog} className="provider-keys-dialog" aria-labelledby="provider-keys-title" aria-describedby="provider-keys-privacy" onCancel={(event) => { event.preventDefault(); onClose(); }}>
      <h2 id="provider-keys-title">{hasKeys ? "API key" : "Add an API key to start"}</h2>
      {!hasKeys && <p>Connect OpenAI for answers and visualization routing. Your message will stay ready to send.</p>}
      <p id="provider-keys-privacy">Your key stays only in this browser’s memory until you refresh or clear it. It is sent through this app’s server to OpenAI. Provider usage is charged to your account.</p>
      <p>Saving or clearing the key starts a new chat.</p>
      <form onSubmit={(event) => { event.preventDefault(); void testConnection(); }}>
        <label htmlFor="provider-openai-key">OpenAI API key</label>
        <input id="provider-openai-key" type="password" maxLength={4096} autoComplete="off" spellCheck={false} autoCapitalize="none" value={openai} onChange={(event) => edit(setOpenai, event.target.value)} disabled={pending || disabled} />
        {error && <p role="alert">{error}</p>}
        <p role="status" aria-live="polite">{pending ? "Checking the provider…" : validated ? "The key works. Save to start a new chat." : hasKeys ? "Your own key is active for this session." : "A key is required to start chatting."}</p>
        <div className="provider-keys-actions">
          {hasKeys && <button type="button" disabled={pending || disabled} onClick={() => { clear(); onClose(); }}>Clear key</button>}
          <button type="button" onClick={onClose}>Cancel</button>
          <button type="submit" disabled={pending || disabled || !openai.trim()}>Test connection</button>
          <button type="button" className="provider-keys-save" disabled={!validated || pending || disabled} onClick={() => { save({ openai: openai.trim() }); onClose(); }}>Save key</button>
        </div>
      </form>
    </dialog>
  );
}
