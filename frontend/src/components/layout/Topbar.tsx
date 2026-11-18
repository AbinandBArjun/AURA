import { Bell, Menu, Search } from "lucide-react";

interface TopbarProps {
  onMenuClick: () => void;
}

export default function Topbar({ onMenuClick }: TopbarProps) {
  return (
    <header className="topbar">
      <button
        className="mobile-menu"
        onClick={onMenuClick}
        aria-label="Open navigation"
      >
        <Menu size={21} />
      </button>

      <div className="topbar-search">
        <Search size={17} />
        <span>Research anything...</span>
        <kbd>Ctrl K</kbd>
      </div>

      <div className="topbar-actions">
        <button
          className="icon-button"
          aria-label="Notifications"
          title="Notifications"
        >
          <Bell size={19} />
        </button>

        <div className="user-avatar">A</div>
      </div>
    </header>
  );
}
