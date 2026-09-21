export type Json = string | number | boolean | null | { [key: string]: Json | undefined } | Json[];

type WorkplaceType = "garage" | "massage";
type ZoneKind = "vehicle_bay" | "tool_area" | "entrance" | "waiting" | "treatment_room" | "reception";
type CameraProtocol = "webcam" | "rtsp" | "phone" | "onvif" | "tapo" | "webrtc";
type ComplaintStatus = "open" | "in_progress" | "resolved";
type ComplaintChannel = "in_app" | "telegram" | "import";

export type Database = {
  public: {
    Tables: {
      cameras: {
        Row: {
          id: string;
          user_id: string;
          external_id: string | null;
          name: string;
          zone: string;
          protocol: CameraProtocol;
          source_url: string;
          main_source_url: string;
          username: string;
          password: string;
          vendor: string;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          external_id?: string | null;
          name: string;
          zone?: string;
          protocol?: CameraProtocol;
          source_url?: string;
          main_source_url?: string;
          username?: string;
          password?: string;
          vendor?: string;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          external_id?: string | null;
          name?: string;
          zone?: string;
          protocol?: CameraProtocol;
          source_url?: string;
          main_source_url?: string;
          username?: string;
          password?: string;
          vendor?: string;
          created_at?: string;
          updated_at?: string;
        };
        Relationships: [];
      };
      crew_identities: {
        Row: {
          id: string;
          user_id: string;
          display_name: string;
          role: string;
          photo_path: string | null;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          display_name: string;
          role?: string;
          photo_path?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          display_name?: string;
          role?: string;
          photo_path?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Relationships: [];
      };
      profiles: {
        Row: {
          id: string;
          display_name: string;
          venue_name: string;
          workplace_type: WorkplaceType;
          avatar_path: string | null;
          setup_completed: boolean;
          telegram_chat_id: string | null;
          telegram_user_id: string | null;
          telegram_username: string | null;
          telegram_linked_at: string | null;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id: string;
          display_name?: string;
          venue_name?: string;
          workplace_type?: WorkplaceType;
          avatar_path?: string | null;
          setup_completed?: boolean;
          telegram_chat_id?: string | null;
          telegram_user_id?: string | null;
          telegram_username?: string | null;
          telegram_linked_at?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          display_name?: string;
          venue_name?: string;
          workplace_type?: WorkplaceType;
          avatar_path?: string | null;
          setup_completed?: boolean;
          telegram_chat_id?: string | null;
          telegram_user_id?: string | null;
          telegram_username?: string | null;
          telegram_linked_at?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Relationships: [];
      };
      roi_bays: {
        Row: {
          id: string;
          user_id: string;
          camera_id: string | null;
          external_id: string | null;
          name: string;
          bay_type: ZoneKind;
          zone_kind: ZoneKind;
          roi: number[];
          sort_order: number;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          camera_id?: string | null;
          external_id?: string | null;
          name: string;
          bay_type?: ZoneKind;
          zone_kind?: ZoneKind;
          roi: number[];
          sort_order?: number;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          camera_id?: string | null;
          external_id?: string | null;
          name?: string;
          bay_type?: ZoneKind;
          zone_kind?: ZoneKind;
          roi?: number[];
          sort_order?: number;
          created_at?: string;
          updated_at?: string;
        };
        Relationships: [];
      };
      anonymous_subjects: {
        Row: {
          id: string;
          user_id: string;
          local_track_key: string;
          first_seen_at: string;
          last_seen_at: string;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          local_track_key: string;
          first_seen_at?: string;
          last_seen_at?: string;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          local_track_key?: string;
          first_seen_at?: string;
          last_seen_at?: string;
          created_at?: string;
          updated_at?: string;
        };
        Relationships: [];
      };
      customer_visits: {
        Row: {
          id: string;
          user_id: string;
          subject_id: string | null;
          zone_id: string;
          started_at: string;
          ended_at: string | null;
          source: "edge" | "import";
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          subject_id?: string | null;
          zone_id?: string;
          started_at?: string;
          ended_at?: string | null;
          source?: "edge" | "import";
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          subject_id?: string | null;
          zone_id?: string;
          started_at?: string;
          ended_at?: string | null;
          source?: "edge" | "import";
          created_at?: string;
          updated_at?: string;
        };
        Relationships: [];
      };
      complaints: {
        Row: {
          id: string;
          user_id: string;
          status: ComplaintStatus;
          channel: ComplaintChannel;
          body: string;
          subject_id: string | null;
          visit_id: string | null;
          external_ref: string | null;
          payload: Json;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          status?: ComplaintStatus;
          channel?: ComplaintChannel;
          body?: string;
          subject_id?: string | null;
          visit_id?: string | null;
          external_ref?: string | null;
          payload?: Json;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          status?: ComplaintStatus;
          channel?: ComplaintChannel;
          body?: string;
          subject_id?: string | null;
          visit_id?: string | null;
          external_ref?: string | null;
          payload?: Json;
          created_at?: string;
          updated_at?: string;
        };
        Relationships: [];
      };
      pipeline_graphs: {
        Row: {
          id: string;
          user_id: string;
          workplace_type: WorkplaceType;
          graph: Json;
          is_active: boolean;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          workplace_type?: WorkplaceType;
          graph?: Json;
          is_active?: boolean;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          workplace_type?: WorkplaceType;
          graph?: Json;
          is_active?: boolean;
          created_at?: string;
          updated_at?: string;
        };
        Relationships: [];
      };
    };
    Views: Record<string, never>;
    Functions: {
      replace_telegram_link: {
        Args: {
          p_user_id: string;
          p_chat_id: string;
          p_telegram_user_id?: string | null;
          p_username?: string | null;
        };
        Returns: undefined;
      };
    };
    Enums: Record<string, never>;
    CompositeTypes: Record<string, never>;
  };
};
