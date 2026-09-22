import { useEffect, useRef } from "react";
import { formatClock } from "../format";
import { useOps } from "../store";

export function TelegramPanel() {
  const { state, telegramSubmitTicket, telegramCheckStatus } = useOps();
  const messages = [...state.messages].sort((a, b) => a.ts - b.ts);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = logRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [messages.length]);

  return (
    <aside className="chat" aria-label="Telegram bot">
      <header className="chat__head">
        <strong>Aeon Mall Security</strong>
        <span>getMe ok · getUpdates polling</span>
      </header>
      <div className="chat__log" ref={logRef}>
        {messages.map((message) => (
          <div key={message.id} className={`bubble is-${message.direction}`}>
            <div>{message.body}</div>
            {message.json && <pre>{message.json}</pre>}
            <time dateTime={new Date(message.ts).toISOString()}>
              {message.method} · {formatClock(message.ts)}
            </time>
          </div>
        ))}
      </div>
      <div className="chat__actions">
        <button className="btn btn--primary" type="button" onClick={telegramSubmitTicket}>
          Submit Ticket
        </button>
        <button className="btn btn--ghost" type="button" onClick={telegramCheckStatus}>
          Check Status
        </button>
      </div>
    </aside>
  );
}
