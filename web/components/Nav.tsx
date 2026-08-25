"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { href: "/", label: "Dashboard" },
  { href: "/specs", label: "Specs" },
  { href: "/run", label: "Run" },
] as const;

export function Nav() {
  const pathname = usePathname();

  return (
    <nav className="nav" role="navigation" aria-label="Main navigation">
      <Link href="/" className="nav-logo" aria-label="ScholarDevClaw Home">
        ScholarDev<span>Claw</span>
      </Link>
      <ul className="nav-links">
        {navItems.map((item) => (
          <li key={item.href}>
            <Link
              href={item.href}
              className={pathname === item.href || pathname.startsWith(item.href + "/") ? "active" : ""}
              aria-current={pathname === item.href ? "page" : undefined}
            >
              {item.label}
            </Link>
          </li>
        ))}
      </ul>
      <div className="nav-status" id="nav-status">
        <span className="badge badge-idle" id="server-status">
          <span className="dot"></span>Server
        </span>
      </div>
    </nav>
  );
}