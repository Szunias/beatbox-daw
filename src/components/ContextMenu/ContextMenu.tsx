/**
 * ContextMenu Component
 * Shows context menu on right-click (desktop) or two-finger tap / long press (mobile)
 */

import React, { useEffect, useRef } from 'react';

export interface MenuItem {
  label: string;
  onClick: () => void;
  icon?: string;
  disabled?: boolean;
  divider?: boolean;
}

interface ContextMenuProps {
  x: number;
  y: number;
  items: MenuItem[];
  onClose: () => void;
}

export const ContextMenu: React.FC<ContextMenuProps> = ({ x, y, items, onClose }) => {
  const menuRef = useRef<HTMLDivElement>(null);

  // Close on click outside or escape
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent | TouchEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    };

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('touchstart', handleClickOutside);
    document.addEventListener('keydown', handleEscape);

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('touchstart', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [onClose]);

  // Adjust position to keep menu in viewport
  useEffect(() => {
    if (menuRef.current) {
      const rect = menuRef.current.getBoundingClientRect();
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;

      if (rect.right > viewportWidth) {
        menuRef.current.style.left = `${x - rect.width}px`;
      }
      if (rect.bottom > viewportHeight) {
        menuRef.current.style.top = `${y - rect.height}px`;
      }
    }
  }, [x, y]);

  return (
    <div
      ref={menuRef}
      className="context-menu"
      style={{
        position: 'fixed',
        left: x,
        top: y,
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid rgba(255,255,255,0.15)',
        borderRadius: 8,
        padding: '6px 0',
        minWidth: 160,
        boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
        zIndex: 9999,
      }}
    >
      {items.map((item, index) => {
        if (item.divider) {
          return (
            <div
              key={index}
              style={{
                height: 1,
                backgroundColor: 'rgba(255,255,255,0.1)',
                margin: '6px 0',
              }}
            />
          );
        }

        return (
          <button
            key={index}
            onClick={() => {
              if (!item.disabled) {
                item.onClick();
                onClose();
              }
            }}
            disabled={item.disabled}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              width: '100%',
              padding: '10px 16px',
              border: 'none',
              background: 'transparent',
              color: item.disabled ? 'var(--text-secondary)' : 'var(--text-primary)',
              fontSize: '0.9rem',
              textAlign: 'left',
              cursor: item.disabled ? 'default' : 'pointer',
              opacity: item.disabled ? 0.5 : 1,
            }}
            onMouseEnter={(e) => {
              if (!item.disabled) {
                e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.1)';
              }
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
            }}
          >
            {item.icon && <span style={{ width: 18, textAlign: 'center' }}>{item.icon}</span>}
            {item.label}
          </button>
        );
      })}
    </div>
  );
};

export default ContextMenu;
