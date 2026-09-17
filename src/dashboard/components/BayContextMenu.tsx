type Props = {
  open: boolean;
  x: number;
  y: number;
  zoneNoun?: string;
  toggleLabel?: string;
  onRename: () => void;
  onToggleType: () => void;
  onDelete: () => void;
};

export function BayContextMenu({
  open,
  x,
  y,
  zoneNoun = "bay",
  toggleLabel = "Vehicle Bay / Tool Station",
  onRename,
  onToggleType,
  onDelete,
}: Props) {
  if (!open) return null;
  const noun = zoneNoun.charAt(0).toUpperCase() + zoneNoun.slice(1);
  return (
    <div
      id="bay-context-menu"
      className="bay-ctx"
      style={{ left: x, top: y }}
      role="menu"
    >
      <button type="button" onClick={onRename}>
        <span>✏️</span> Rename {noun}
      </button>
      <button type="button" onClick={onToggleType}>
        <span>🏷️</span> Toggle {toggleLabel}
      </button>
      <div className="bay-ctx__rule" />
      <button type="button" className="is-danger" onClick={onDelete}>
        <span>🗑️</span> Delete {noun}
      </button>
    </div>
  );
}
