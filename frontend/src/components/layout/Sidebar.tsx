import {
  Activity,
  BrainCircuit,
  FileText,
  FolderOpen,
  LayoutDashboard,
  Search,
  X,
} from "lucide-react";
import { NavLink } from "react-router-dom";

interface SidebarProps {
  isOpen: boolean;
  onNavigate: () => void;
}

const navigation = [
  {
    label: "OVERVIEW",
    items: [
      { name: "Dashboard", path: "/", icon: LayoutDashboard },
    ],
  },
  {
    label: "WORKSPACE",
    items: [
      { name: "Documents", path: "/documents", icon: FolderOpen },
      { name: "Knowledge Search", path: "/search", icon: Search },
    ],
  },
  {
    label: "INTELLIGENCE",
    items: [
      { name: "Research Agents", path: "/research", icon: BrainCircuit },
      { name: "Reports", path: "/reports", icon: FileText },
      { name: "Analytics", path: "/analytics", icon: Activity },
    ],
  },
];

export default function Sidebar({
  isOpen,
  onNavigate,
}: SidebarProps) {
  return (
    <>
      {isOpen && (
        <button
          className="sidebar-overlay"
          onClick={onNavigate}
          aria-label="Close navigation"
        />
      )}

      <aside className={`sidebar ${isOpen ? "sidebar-open" : ""}`}>
        <div className="sidebar-brand">
          <div className="brand-symbol">
            <BrainCircuit size={23} />
          </div>

          <div>
            <h2>AURA</h2>
            <span>RESEARCH INTELLIGENCE</span>
          </div>

          <button
            className="mobile-close"
            onClick={onNavigate}
            aria-label="Close sidebar"
          >
            <X size={19} />
          </button>
        </div>

        <div className="sidebar-content">
          {navigation.map((group) => (
            <div className="nav-group" key={group.label}>
              <p className="nav-label">{group.label}</p>

              {group.items.map((item) => {
                const Icon = item.icon;

                return (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    end={item.path === "/"}
                    className={({ isActive }) =>
                      `nav-item ${isActive ? "nav-item-active" : ""}`
                    }
                    onClick={onNavigate}
                  >
                    <Icon size={18} strokeWidth={1.8} />
                    <span>{item.name}</span>
                  </NavLink>
                );
              })}
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <div className="sidebar-footer-icon">
            <BrainCircuit size={18} />
          </div>
          <div>
            <strong>AURA Platform</strong>
            <span>Version 0.1.0</span>
          </div>
        </div>
      </aside>
    </>
  );
}
