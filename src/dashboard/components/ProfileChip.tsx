import { useEffect, useRef, useState } from "react";
import { initials } from "../account";
import { useAccount } from "../auth";

export function ProfileChip({ onOpenSetup }: { onOpenSetup: () => void }) {
  const { snapshot, user, signOut } = useAccount();
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const profile = snapshot?.profile;
  const name = profile?.display_name || user?.email?.split("@")[0] || "Operator";
  const venue = profile?.venue_name || "Private operator";

  useEffect(() => {
    function onDown(event: MouseEvent) {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    }
    window.addEventListener("mousedown", onDown);
    return () => window.removeEventListener("mousedown", onDown);
  }, []);

  return (
    <div className="profile-chip" ref={root}>
      <button className="profile-chip__btn" type="button" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        {snapshot?.avatarUrl ? (
          <img className="profile-chip__avatar" src={snapshot.avatarUrl} alt="" />
        ) : (
          <span className="profile-chip__avatar is-fallback">{initials(name, user?.email)}</span>
        )}
        <span className="profile-chip__copy">
          <strong>{name}</strong>
          <span>{venue}</span>
        </span>
      </button>
      {open ? (
        <div className="profile-menu" role="menu">
          <p className="profile-menu__email">{user?.email}</p>
          <button
            type="button"
            onClick={() => {
              setOpen(false);
              onOpenSetup();
            }}
          >
            Account setup
          </button>
          <button
            type="button"
            onClick={() => {
              setOpen(false);
              void signOut();
            }}
          >
            Sign out
          </button>
        </div>
      ) : null}
    </div>
  );
}
