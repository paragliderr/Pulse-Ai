"use client";
import Link from "next/link";
import { useEffect, useState } from "react";

export default function Nav() {
  const [loggedIn, setLoggedIn] = useState(false);
  useEffect(() => setLoggedIn(!!localStorage.getItem("pulse_token")), []);

  function logout() {
    localStorage.removeItem("pulse_token");
    window.location.href = "/";
  }

  return (
    <nav className="mx-auto flex max-w-3xl items-center justify-between px-4 py-5 text-sm text-white/70">
      <Link href="/" className="font-display text-xl italic text-white">Pulse</Link>
      <div className="flex gap-5">
        <Link href="/dashboard" className="hover:text-white">Feed</Link>
        {loggedIn && <Link href="/watchlist" className="hover:text-white">Watchlist</Link>}
        {loggedIn ? (
          <button onClick={logout} className="hover:text-white">Log out</button>
        ) : (
          <>
            <Link href="/login" className="hover:text-white">Log in</Link>
            <Link href="/signup" className="text-brass">Sign up</Link>
          </>
        )}
      </div>
    </nav>
  );
}
