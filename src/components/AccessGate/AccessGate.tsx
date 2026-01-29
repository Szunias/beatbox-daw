/**
 * AccessGate Component
 * Early access screen requiring access code to unlock the application
 */

import React, { useState, useCallback } from 'react';
import { useAccessCode } from '../../hooks/useAccessCode';

interface AccessGateProps {
  children: React.ReactNode;
}

export const AccessGate: React.FC<AccessGateProps> = ({ children }) => {
  const { isUnlocked, validateCode } = useAccessCode();
  const [code, setCode] = useState('');
  const [error, setError] = useState(false);
  const [shake, setShake] = useState(false);

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();

    if (validateCode(code)) {
      setError(false);
    } else {
      setError(true);
      setShake(true);
      setTimeout(() => setShake(false), 500);
    }
  }, [code, validateCode]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSubmit(e);
    }
  }, [handleSubmit]);

  if (isUnlocked) {
    return <>{children}</>;
  }

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <div style={styles.logo}>
          <div style={styles.logoIcon}>
            <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
              <circle cx="24" cy="24" r="20" stroke="var(--accent-primary)" strokeWidth="3" />
              <circle cx="24" cy="24" r="8" fill="var(--accent-primary)" />
              <path d="M24 4V12" stroke="var(--accent-primary)" strokeWidth="2" strokeLinecap="round" />
              <path d="M24 36V44" stroke="var(--accent-primary)" strokeWidth="2" strokeLinecap="round" />
              <path d="M4 24H12" stroke="var(--accent-primary)" strokeWidth="2" strokeLinecap="round" />
              <path d="M36 24H44" stroke="var(--accent-primary)" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </div>
          <h1 style={styles.title}>BeatBox DAW</h1>
        </div>

        <div style={styles.badge}>Early Access</div>

        <p style={styles.description}>
          Real-time beatbox to MIDI conversion powered by machine learning.
          Enter your access code to continue.
        </p>

        <form onSubmit={handleSubmit} style={styles.form}>
          <input
            type="text"
            value={code}
            onChange={(e) => {
              setCode(e.target.value);
              setError(false);
            }}
            onKeyDown={handleKeyDown}
            placeholder="Enter access code"
            style={{
              ...styles.input,
              ...(error ? styles.inputError : {}),
              ...(shake ? styles.shake : {}),
            }}
            autoFocus
          />
          <button type="submit" style={styles.button}>
            Unlock
          </button>
        </form>

        {error && (
          <p style={styles.errorText}>Invalid access code. Please try again.</p>
        )}

        <div style={styles.footer}>
          <p style={styles.footerText}>
            Request access or download the desktop app at{' '}
            <a
              href="https://github.com/yourusername/beatbox-daw"
              target="_blank"
              rel="noopener noreferrer"
              style={styles.link}
            >
              github.com/beatbox-daw
            </a>
          </p>
        </div>
      </div>

      <style>{`
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          25% { transform: translateX(-8px); }
          75% { transform: translateX(8px); }
        }
      `}</style>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '20px',
    background: 'linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%)',
  },
  card: {
    width: '100%',
    maxWidth: '420px',
    padding: '40px',
    backgroundColor: 'var(--bg-secondary)',
    borderRadius: '16px',
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
    textAlign: 'center',
  },
  logo: {
    marginBottom: '24px',
  },
  logoIcon: {
    marginBottom: '16px',
  },
  title: {
    fontSize: '1.8rem',
    fontWeight: 700,
    background: 'linear-gradient(90deg, var(--accent-primary), #ff6b6b)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
    margin: 0,
  },
  badge: {
    display: 'inline-block',
    padding: '6px 16px',
    backgroundColor: 'rgba(233, 69, 96, 0.15)',
    color: 'var(--accent-primary)',
    borderRadius: '20px',
    fontSize: '0.85rem',
    fontWeight: 600,
    marginBottom: '20px',
    border: '1px solid rgba(233, 69, 96, 0.3)',
  },
  description: {
    color: 'var(--text-secondary)',
    fontSize: '0.95rem',
    lineHeight: 1.6,
    marginBottom: '28px',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  input: {
    width: '100%',
    padding: '14px 18px',
    fontSize: '1rem',
    border: '2px solid var(--bg-tertiary)',
    borderRadius: '10px',
    backgroundColor: 'var(--bg-primary)',
    color: 'var(--text-primary)',
    outline: 'none',
    transition: 'border-color 0.2s, box-shadow 0.2s',
    textAlign: 'center',
    letterSpacing: '2px',
    textTransform: 'uppercase',
  },
  inputError: {
    borderColor: 'var(--error)',
    boxShadow: '0 0 0 3px rgba(239, 68, 68, 0.2)',
  },
  shake: {
    animation: 'shake 0.5s ease-in-out',
  },
  button: {
    width: '100%',
    padding: '14px',
    fontSize: '1rem',
    fontWeight: 600,
    border: 'none',
    borderRadius: '10px',
    backgroundColor: 'var(--accent-primary)',
    color: 'white',
    cursor: 'pointer',
    transition: 'transform 0.15s, box-shadow 0.15s',
  },
  errorText: {
    color: 'var(--error)',
    fontSize: '0.85rem',
    marginTop: '12px',
  },
  footer: {
    marginTop: '32px',
    paddingTop: '20px',
    borderTop: '1px solid var(--bg-tertiary)',
  },
  footerText: {
    color: 'var(--text-secondary)',
    fontSize: '0.8rem',
    margin: 0,
  },
  link: {
    color: 'var(--accent-primary)',
    textDecoration: 'none',
  },
};

export default AccessGate;
