"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import api from "../lib/api";

export default function AuthForm({ signup }) {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);

  async function submit(e) {
    e.preventDefault();
    try {
      const res = await api.post(signup ? "/api/auth/signup" : "/api/auth/login", { email, password });
      localStorage.setItem("pulse_token", res.data.token);
      router.push(signup ? "/watchlist" : "/dashboard"); // new users pick tickers first
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    }
  }

  const input = "rounded-lg border border-white/15 bg-white/5 p-3 text-white placeholder-white/30 outline-none focus:border-brass";
  return (
    <main className="mx-auto mt-24 max-w-sm px-4 text-white">
      <Link href="/" className="font-display text-2xl italic">Pulse</Link>
      <h1 className="mt-6 font-display text-3xl">{signup ? "Create your account" : "Welcome back"}</h1>
      <form onSubmit={submit} className="mt-6 flex flex-col gap-3">
        <input className={input} type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        <input className={input} type="password" placeholder="Password (min 8 characters)" value={password} onChange={(e) => setPassword(e.target.value)} required />
        {error && <p className="text-sm text-signal-red">{error}</p>}
        <button className="rounded-full bg-brass p-3 font-medium text-ink">{signup ? "Sign up" : "Log in"}</button>
      </form>
      <p className="mt-5 text-sm text-white/60">
        {signup ? <>Have an account? <Link href="/login" className="text-brass">Log in</Link></>
                : <>New here? <Link href="/signup" className="text-brass">Sign up</Link></>}
      </p>
      <p className="mt-2 text-sm text-white/60"><Link href="/dashboard" className="underline">Skip - see the demo feed</Link></p>
    </main>
  );
}
