-- Pipeline graph IR per operator. Source of truth for the node editor.
-- Owner-only RLS. workplace_type is a row field, not a JWT user_metadata claim.

create table if not exists public.pipeline_graphs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  workplace_type text not null default 'garage',
  graph jsonb not null default '{}'::jsonb,
  is_active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint pipeline_graphs_workplace_check check (workplace_type in ('garage', 'massage'))
);

create unique index if not exists pipeline_graphs_one_active_uidx
  on public.pipeline_graphs (user_id)
  where is_active;

create index if not exists pipeline_graphs_user_id_idx on public.pipeline_graphs (user_id);

comment on table public.pipeline_graphs is
  'Typed pipeline graph IR. Editor saves JSON; edge compiler emits engine config.';

alter table public.pipeline_graphs enable row level security;

revoke all on table public.pipeline_graphs from anon, public;
grant select, insert, update, delete on table public.pipeline_graphs to authenticated;

drop policy if exists pipeline_graphs_select_own on public.pipeline_graphs;
drop policy if exists pipeline_graphs_insert_own on public.pipeline_graphs;
drop policy if exists pipeline_graphs_update_own on public.pipeline_graphs;
drop policy if exists pipeline_graphs_delete_own on public.pipeline_graphs;
create policy pipeline_graphs_select_own on public.pipeline_graphs
  for select to authenticated
  using (user_id = (select auth.uid()));
create policy pipeline_graphs_insert_own on public.pipeline_graphs
  for insert to authenticated
  with check (user_id = (select auth.uid()));
create policy pipeline_graphs_update_own on public.pipeline_graphs
  for update to authenticated
  using (user_id = (select auth.uid()))
  with check (user_id = (select auth.uid()));
create policy pipeline_graphs_delete_own on public.pipeline_graphs
  for delete to authenticated
  using (user_id = (select auth.uid()));

drop trigger if exists pipeline_graphs_set_updated_at on public.pipeline_graphs;
create trigger pipeline_graphs_set_updated_at
  before update on public.pipeline_graphs
  for each row execute function private.set_updated_at();
