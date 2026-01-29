import React from 'react';

interface RecordButtonProps {
  isRecording: boolean;
  isConnected: boolean;
  onToggle: () => void;
}

export const RecordButton: React.FC<RecordButtonProps> = ({
  isRecording,
  isConnected,
  onToggle,
}) => {
  return (
    <button
      className={`record-button ${isRecording ? 'recording' : 'idle'}`}
      onClick={onToggle}
      disabled={!isConnected}
      title={!isConnected ? 'Connect to engine first' : undefined}
    >
      <span className="record-icon" />
      {isRecording ? 'Stop Recording' : 'Start Recording'}
    </button>
  );
};
