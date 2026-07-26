import React, { useState, useRef, useEffect } from 'react';
import { chatWithInspector } from '../services/api';

function ChatInspector({ onNavigate, lastRecommendation }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const send = async (e) => {
    e.preventDefault();
    if (!input.trim() || !lastRecommendation) return;
    const text = input;
    setMessages(p => [...p, { role: 'user', content: text }]);
    setInput('');
    setLoading(true);
    try {
      const r = await chatWithInspector({ query: text, context: lastRecommendation });
      setMessages(p => [...p, { role: 'bot', content: r.response }]);
    } catch {
      setMessages(p => [...p, { role: 'bot', content: 'Error processing your question. Please try again.' }]);
    }
    setLoading(false);
  };

  const suggestions = [
    "What are the main structural concerns?",
    "What repairs should be prioritized?",
    "Is the building safe for occupancy?",
    "Estimated cost of repairs?",
  ];

  if (!lastRecommendation) {
    return (
      <div className="anim-fade-up">
        <div className="pg-header">
          <h1>Chat Inspector</h1>
          <p>Ask questions about your structural analysis</p>
        </div>
        <div className="g-card">
          <div className="empty">
            <h4>No inspection context</h4>
            <p style={{ marginBottom: 16 }}>Run Risk Predictor first to generate context for the chat.</p>
            <button className="btn btn-primary" onClick={() => onNavigate('risk')}>Go to Risk Predictor</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="anim-fade-up">
      <div className="pg-header">
        <h1>Chat Inspector</h1>
        <p>Ask questions about your structural analysis</p>
      </div>

      <div className="chat-wrap" style={{ maxWidth: 780, minHeight: 500 }}>
        <div className="chat-top">
          Structural Health AI Inspector
          <span className="m-pill safe" style={{ marginLeft: 'auto', fontSize: 11 }}>Online</span>
        </div>

        <div className="chat-body" style={{ flex: 1 }}>
          {messages.length === 0 ? (
            <div className="chat-empty">
              <div style={{ fontSize: 14, color: 'var(--text-3)', maxWidth: 300, textAlign: 'center' }}>
                Ask any question about the inspection report.
              </div>
              <div className="chat-suggestions">
                {suggestions.map((s, i) => (
                  <button key={i} className="chat-suggest-btn" onClick={() => setInput(s)}>{s}</button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((m, i) => (
              <div key={i} className={`chat-bubble ${m.role}`}>{m.content}</div>
            ))
          )}
          {loading && <div className="chat-thinking">Thinking...</div>}
          <div ref={endRef} />
        </div>

        <form onSubmit={send} className="chat-bar">
          <input
            className="chat-field"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask about the inspection..."
            disabled={loading}
          />
          <button type="submit" className="btn btn-primary" disabled={loading || !input.trim()}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default ChatInspector;
