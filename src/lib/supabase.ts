import { createClient } from "@supabase/supabase-js";
import type { Database } from "./database.types";

const DEFAULT_SUPABASE_URL = "https://rmepwjywobowktdmkusu.supabase.co";
const DEFAULT_SUPABASE_ANON_KEY =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJtZXB3anl3b2Jvd2t0ZG1rdXN1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODg3NDA0MDYsImV4cCI6MjEwNDMxNjQwNn0.u_hg9uXPSK7AE1kD3ol9LUmYgyddVWrELACWh5Z9zr4";

export const SUPABASE_URL = (
  import.meta.env.VITE_SUPABASE_URL ||
  DEFAULT_SUPABASE_URL
).trim();
export const SUPABASE_ANON_KEY = (
  import.meta.env.VITE_SUPABASE_ANON_KEY ||
  import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY ||
  DEFAULT_SUPABASE_ANON_KEY
).trim();

export const supabaseConfigured = Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);
export const ACCOUNT_BUCKET = "account-private";

export const supabase = createClient<Database>(
  SUPABASE_URL || "https://unavailable.supabase.co",
  SUPABASE_ANON_KEY || "missing-anon-key",
  {
    auth: {
      persistSession: true,
      autoRefreshToken: true,
      detectSessionInUrl: true,
      flowType: "pkce",
    },
  },
);
