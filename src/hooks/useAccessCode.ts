/**
 * Access Code Hook
 * Manages early access code validation and persistence
 */

import { useState, useCallback, useEffect } from 'react';

const STORAGE_KEY = 'beatbox-daw-access';
const VALID_CODES = ['BEATBOX2024', 'EARLYACCESS', 'DEMO'];

interface UseAccessCodeReturn {
  isUnlocked: boolean;
  validateCode: (code: string) => boolean;
  reset: () => void;
}

export function useAccessCode(): UseAccessCodeReturn {
  const [isUnlocked, setIsUnlocked] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    return localStorage.getItem(STORAGE_KEY) === 'unlocked';
  });

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'unlocked') {
      setIsUnlocked(true);
    }
  }, []);

  const validateCode = useCallback((code: string): boolean => {
    const normalizedCode = code.trim().toUpperCase();
    const isValid = VALID_CODES.includes(normalizedCode);

    if (isValid) {
      localStorage.setItem(STORAGE_KEY, 'unlocked');
      setIsUnlocked(true);
    }

    return isValid;
  }, []);

  const reset = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setIsUnlocked(false);
  }, []);

  return {
    isUnlocked,
    validateCode,
    reset,
  };
}
