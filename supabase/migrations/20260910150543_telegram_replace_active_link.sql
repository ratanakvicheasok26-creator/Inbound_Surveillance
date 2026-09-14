-- One active Telegram per operator profile.
-- Connecting a new Telegram overwrites telegram_chat_id. Alerts must go only
-- to that latest chat — never to a previously linked Telegram.

-- If a chat id was stored as CSV / JSON-ish list, keep only the last value.
update public.profiles
set telegram_chat_id = nullif(
  btrim(
    btrim(
      split_part(
        replace(replace(coalesce(telegram_chat_id, ''), '[', ''), ']', ''),
        ',',
        greatest(
          1,
          coalesce(
            array_length(
              string_to_array(
                replace(replace(coalesce(telegram_chat_id, ''), '[', ''), ']', ''),
                ','
              ),
              1
            ),
            1
          )
        )
      )
    ),
    '"'
  ),
  ''
)
where telegram_chat_id is not null
  and (
    telegram_chat_id like '%,%'
    or telegram_chat_id like '[%'
  );

create or replace function private.replace_telegram_link(
  p_chat_id text,
  p_telegram_user_id text default null,
  p_username text default null
)
returns public.profiles
language plpgsql
security definer
set search_path = ''
as $$
declare
  uid uuid := (select auth.uid());
  result public.profiles;
  chat text := nullif(trim(p_chat_id), '');
begin
  if uid is null then
    raise exception 'Not authenticated';
  end if;
  if chat is null then
    raise exception 'chat_id required';
  end if;

  -- Unique index on telegram_chat_id: free this chat from any other profile.
  update public.profiles
     set telegram_chat_id = null,
         telegram_user_id = null,
         telegram_username = null,
         telegram_linked_at = null
   where telegram_chat_id = chat
     and id <> uid;

  update public.profiles
     set telegram_chat_id = chat,
         telegram_user_id = nullif(trim(coalesce(p_telegram_user_id, '')), ''),
         telegram_username = nullif(trim(coalesce(p_username, '')), ''),
         telegram_linked_at = now()
   where id = uid
  returning * into result;

  if result.id is null then
    raise exception 'Profile not found';
  end if;
  return result;
end;
$$;

revoke all on function private.replace_telegram_link(text, text, text)
  from public, anon, authenticated;

create or replace function public.replace_telegram_link(
  p_chat_id text,
  p_telegram_user_id text default null,
  p_username text default null
)
returns public.profiles
language plpgsql
security definer
set search_path = ''
as $$
begin
  return private.replace_telegram_link(p_chat_id, p_telegram_user_id, p_username);
end;
$$;

revoke all on function public.replace_telegram_link(text, text, text)
  from public, anon;
grant execute on function public.replace_telegram_link(text, text, text)
  to authenticated;

comment on function public.replace_telegram_link(text, text, text) is
  'Replace this operator''s Telegram destination. Previous chat_id is overwritten; alerts go only to the latest chat.';
