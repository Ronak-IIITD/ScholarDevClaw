import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  BookOpen,
  Play,
  Github,
} from "lucide-react";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/specs", label: "Paper Specs", icon: BookOpen },
  { to: "/pipeline", label: "Pipeline", icon: Play },
] as const;

export default function Sidebar() {
  return (
    <aside className="flex w-64 flex-col border-r border-gray-800 bg-gray-900">
      {/* Brand */}
      <div className="flex items-center gap-3 px-6 py-5 border-b border-gray-800">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-600 text-white font-bold text-sm">
          SD
        </div>
        <div>
          <h1 className="text-sm font-semibold text-white">ScholarDevClaw</h1>
          <p className="text-xs text-gray-400">Research-to-Code</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                isActive
                  ? "bg-brand-600/10 text-brand-400"
                  : "text-gray-400 hover:bg-gray-800 hover:text-gray-200"
              }`
            }
          >
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="border-t border-gray-800 px-4 py-4">
        <a
          href="https://github.com/Ronak-IIITD/ScholarDevClaw"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-xs text-gray-500 hover:text-gray-300 transition-colors"
        >
          <Github size={14} />
          View on GitHub
        </a>
        <p className="mt-2 text-xs text-gray-600">v0.1.0</p>
      </div>
    </aside>
  );
}
