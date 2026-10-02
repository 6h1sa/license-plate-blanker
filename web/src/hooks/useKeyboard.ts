import { useEffect, useRef } from "react";

/**
 * キーと処理の対応を登録する。キーは KeyboardEvent.key を小文字にしたもの（" " は Space）。
 * 文字を入力する欄にフォーカスがあるときは何もしない。
 */
export function useKeyboard(handlers: Record<string, () => void>) {
  const ref = useRef(handlers);
  ref.current = handlers;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      const typing = t.tagName === "TEXTAREA" || t.tagName === "SELECT" ||
        (t.tagName === "INPUT" && !["checkbox", "range", "radio"].includes((t as HTMLInputElement).type));
      if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
      const fn = ref.current[e.key.toLowerCase()];
      if (fn) {
        e.preventDefault();
        fn();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}
