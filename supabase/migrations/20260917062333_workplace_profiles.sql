-- Workplace-typed operator accounts, generalized ROI kinds, anonymous
-- customer-visit persistence, and complaint ingest structure.
-- Owner-only RLS. Never authorize from user_metadata.

alter table public.profiles
  add column if not exists workplace_type text not null default 'garage';

alter table public.profiles
  drop constraint if exists profiles_workplace_type_check;

alter table public.profiles
  add constraint profiles_workplace_type_check
  check (workplace_type in ('garage', 'massage'));

comment on column public.profiles.workplace_type is
  'Operator workplace profile. Stored on profiles, never used from user_metadata in RLS.';

alter table public.roi_bays
  add column if not exists zone_kind text;

update public.roi_bays
  set zone_kind = coalesce(nullif(zone_kind, ''), bay_type, 'vehicle_bay')
  where zone_kind is null or zone_kind = '';

alter table public.roi_bays
  alter column zone_kind set default 'vehicle_bay';

alter table public.roi_bays
  alter column zone_kind set not null;

alter table public.roi_bays
  drop constraint if exists roi_bays_type_check;

alter table public.roi_bays
  add constraint roi_bays_type_check
  check (bay_type in ('vehicle_bay', 'tool_area', 'entrance', 'waiting', 'treatment_room'));

alter table public.roi_bays
  drop constraint if exists roi_bays_zone_kind_check;

alter table public.roi_bays
  add constraint roi_bays_zone_kind_check
  check (zone_kind in ('vehicle_bay', 'tool_area', 'entrance', 'waiting', 'treatment_room'));

create or replace function private.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
declare
  meta_name text;
  meta_venue text;
  meta_workplace text;
begin
  meta_name := coalesce(
    nullif(trim(new.raw_user_meta_data ->> 'display_name'), ''),
    nullif(split_part(coalesce(new.email, ''), '@', 1), ''),
    'Operator'
  );
  meta_venue := coalesce(nullif(trim(new.raw_user_meta_data ->> 'venue_name'), ''), '');
  meta_workplace := lower(coalesce(nullif(trim(new.raw_user_meta_data ->> 'workplace_type'), ''), 'garage'));
  if meta_workplace not in ('garage', 'massage') then
    meta_workplace := 'garage';
  end if;
  insert into public.profiles (id, display_name, venue_name, workplace_type)
  values (new.id, meta_name, meta_venue, meta_workplace)
  on conflict (id) do nothing;
  return new;
end;
$$;

create table if not exists public.anonymous_subjects (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  local_track_key text not null,
  first_seen_at timestamptz not null default now(),
  last_seen_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id, local_track_key)
);

create table if not exists public.customer_visits (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  subject_id uuid references public.anonymous_subjects (id) on delete set null,
  zone_id text not null default '',
  started_at timestamptz not null default now(),
  ended_at timestamptz,
  source text not null default 'edge',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint customer_visits_source_check check (source in ('edge', 'import'))
);

create table if not exists public.complaints (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  status text not null default 'open',
  channel text not null default 'import',
  body text not null default '',
  subject_id uuid references public.anonymous_subjects (id) on delete set null,
  visit_id uuid references public.customer_visits (id) on delete set null,
  external_ref text,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint complaints_status_check check (status in ('open', 'in_progress', 'resolved')),
  constraint complaints_channel_check check (channel in ('in_app', 'telegram', 'import'))
);

create unique index if not exists complaints_user_external_ref_uidx
  on public.complaints (user_id, external_ref)
  where external_ref is not null;

create index if not exists anonymous_subjects_user_id_idx on public.anonymous_subjects (user_id);
create index if not exists customer_visits_user_id_idx on public.customer_visits (user_id);
create index if not exists customer_visits_started_at_idx on public.customer_visits (user_id, started_at desc);
create index if not exists complaints_user_id_idx on public.complaints (user_id);

comment on table public.anonymous_subjects is 'Opaque anonymous customer identities. No name or photo.';
comment on table public.customer_visits is 'Customer visit sessions (day/week unique counts).';
comment on table public.complaints is 'Complaint structure only. Capture UI is implemented by another team.';

alter table public.anonymous_subjects enable row level security;
alter table public.customer_visits enable row level security;
alter table public.complaints enable row level security;

revoke all on table public.anonymous_subjects from anon, public;
revoke all on table public.customer_visits from anon, public;
revoke all on table public.complaints from anon, public;

grant select, insert, update, delete on table public.anonymous_subjects to authenticated;
grant select, insert, update, delete on table public.customer_visits to authenticated;
grant select, insert, update, delete on table public.complaints to authenticated;

drop policy if exists anonymous_subjects_select_own on public.anonymous_subjects;
drop policy if exists anonymous_subjects_insert_own on public.anonymous_subjects;
drop policy if exists anonymous_subjects_update_own on public.anonymous_subjects;
drop policy if exists anonymous_subjects_delete_own on public.anonymous_subjects;
create policy anonymous_subjects_select_own on public.anonymous_subjects
  for select to authenticated
  using (user_id = (select auth.uid()));
create policy anonymous_subjects_insert_own on public.anonymous_subjects
  for insert to authenticated
  with check (user_id = (select auth.uid()));
create policy anonymous_subjects_update_own on public.anonymous_subjects
  for update to authenticated
  using (user_id = (select auth.uid()))
  with check (user_id = (select auth.uid()));
create policy anonymous_subjects_delete_own on public.anonymous_subjects
  for delete to authenticated
  using (user_id = (select auth.uid()));

drop policy if exists customer_visits_select_own on public.customer_visits;
drop policy if exists customer_visits_insert_own on public.customer_visits;
drop policy if exists customer_visits_update_own on public.customer_visits;
drop policy if exists customer_visits_delete_own on public.customer_visits;
create policy customer_visits_select_own on public.customer_visits
  for select to authenticated
  using (user_id = (select auth.uid()));
create policy customer_visits_insert_own on public.customer_visits
  for insert to authenticated
  with check (user_id = (select auth.uid()));
create policy customer_visits_update_own on public.customer_visits
  for update to authenticated
  using (user_id = (select auth.uid()))
  with check (user_id = (select auth.uid()));
create policy customer_visits_delete_own on public.customer_visits
  for delete to authenticated
  using (user_id = (select auth.uid()));

drop policy if exists complaints_select_own on public.complaints;
drop policy if exists complaints_insert_own on public.complaints;
drop policy if exists complaints_update_own on public.complaints;
drop policy if exists complaints_delete_own on public.complaints;
create policy complaints_select_own on public.complaints
  for select to authenticated
  using (user_id = (select auth.uid()));
create policy complaints_insert_own on public.complaints
  for insert to authenticated
  with check (user_id = (select auth.uid()));
create policy complaints_update_own on public.complaints
  for update to authenticated
  using (user_id = (select auth.uid()))
  with check (user_id = (select auth.uid()));
create policy complaints_delete_own on public.complaints
  for delete to authenticated
  using (user_id = (select auth.uid()));

drop trigger if exists anonymous_subjects_set_updated_at on public.anonymous_subjects;
create trigger anonymous_subjects_set_updated_at
  before update on public.anonymous_subjects
  for each row execute function private.set_updated_at();

drop trigger if exists customer_visits_set_updated_at on public.customer_visits;
create trigger customer_visits_set_updated_at
  before update on public.customer_visits
  for each row execute function private.set_updated_at();

drop trigger if exists complaints_set_updated_at on public.complaints;
create trigger complaints_set_updated_at
  before update on public.complaints
  for each row execute function private.set_updated_at();
