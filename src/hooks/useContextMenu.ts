/**
 * useContextMenu Hook
 * Handles context menu triggers:
 * - Right-click on desktop
 * - Two-finger tap on mobile
 * - Long press on mobile (optional)
 */

import { useState, useCallback, useRef, useEffect } from 'react';

interface ContextMenuState {
  isOpen: boolean;
  x: number;
  y: number;
}

interface UseContextMenuOptions {
  enableLongPress?: boolean;
  longPressDelay?: number;
}

export const useContextMenu = (options: UseContextMenuOptions = {}) => {
  const { enableLongPress = true, longPressDelay = 500 } = options;

  const [menuState, setMenuState] = useState<ContextMenuState>({
    isOpen: false,
    x: 0,
    y: 0,
  });

  const longPressTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const touchStartPos = useRef<{ x: number; y: number } | null>(null);
  const isTwoFingerTouch = useRef(false);

  const openMenu = useCallback((x: number, y: number) => {
    setMenuState({ isOpen: true, x, y });
  }, []);

  const closeMenu = useCallback(() => {
    setMenuState({ isOpen: false, x: 0, y: 0 });
  }, []);

  // Right-click handler
  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    openMenu(e.clientX, e.clientY);
  }, [openMenu]);

  // Touch handlers for two-finger tap and long press
  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    // Two-finger tap
    if (e.touches.length === 2) {
      e.preventDefault();
      isTwoFingerTouch.current = true;
      const touch1 = e.touches[0];
      const touch2 = e.touches[1];
      // Open menu at center of two fingers
      const centerX = (touch1.clientX + touch2.clientX) / 2;
      const centerY = (touch1.clientY + touch2.clientY) / 2;
      openMenu(centerX, centerY);
      return;
    }

    // Long press (single finger)
    if (enableLongPress && e.touches.length === 1) {
      const touch = e.touches[0];
      touchStartPos.current = { x: touch.clientX, y: touch.clientY };

      longPressTimer.current = setTimeout(() => {
        if (touchStartPos.current) {
          openMenu(touchStartPos.current.x, touchStartPos.current.y);
        }
      }, longPressDelay);
    }
  }, [openMenu, enableLongPress, longPressDelay]);

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    // Cancel long press if finger moved
    if (longPressTimer.current && touchStartPos.current && e.touches.length === 1) {
      const touch = e.touches[0];
      const dx = Math.abs(touch.clientX - touchStartPos.current.x);
      const dy = Math.abs(touch.clientY - touchStartPos.current.y);

      if (dx > 10 || dy > 10) {
        clearTimeout(longPressTimer.current);
        longPressTimer.current = null;
      }
    }
  }, []);

  const handleTouchEnd = useCallback(() => {
    if (longPressTimer.current) {
      clearTimeout(longPressTimer.current);
      longPressTimer.current = null;
    }
    touchStartPos.current = null;
    isTwoFingerTouch.current = false;
  }, []);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (longPressTimer.current) {
        clearTimeout(longPressTimer.current);
      }
    };
  }, []);

  return {
    menuState,
    openMenu,
    closeMenu,
    handlers: {
      onContextMenu: handleContextMenu,
      onTouchStart: handleTouchStart,
      onTouchMove: handleTouchMove,
      onTouchEnd: handleTouchEnd,
    },
  };
};
