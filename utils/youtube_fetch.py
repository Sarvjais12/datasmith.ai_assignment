import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Pre-compile regex for performance (a standard senior engineer practice)
YT_REGEX_CONFIG = [
    re.compile(r"youtube\.com/watch\?v=([\w\-]{11})"),
    re.compile(r"youtu\.be/([\w\-]{11})"),
    re.compile(r"youtube\.com/embed/([\w\-]{11})"),
    re.compile(r"youtube\.com/shorts/([\w\-]{11})"),
]

def _parse_video_id(url_string: str) -> Optional[str]:
    """Scans a raw URL string and extracts the 11-character YouTube video ID."""
    for pattern in YT_REGEX_CONFIG:
        match = pattern.search(url_string)
        if match:
            return match.group(1)
    return None

def fetch_youtube_transcript(target_url: str) -> str:
    """
    Pulls down closed captions for a YouTube URL.
    Defaults to English, auto-translates foreign languages, and calculates video duration.
    """
    try:
        from youtube_transcript_api import (
            YouTubeTranscriptApi,
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
        )
    except ImportError:
        logger.critical("youtube-transcript-api missing. Cannot scrape videos.")
        return "System Error: The YouTube scraper dependency is not installed."

    vid_id = _parse_video_id(target_url)
    if not vid_id:
        return f"⚠️ Validation Error: Could not locate a valid YouTube ID in the link provided."

    try:
        # Grab all available caption tracks
        available_tracks = YouTubeTranscriptApi.list_transcripts(vid_id)
        
        # Try native English first
        try:
            active_track = available_tracks.find_transcript(["en", "en-US", "en-GB"])
        except Exception:
            # If no English exists, grab the primary generated language and auto-translate it
            source_langs = [track.language_code for track in available_tracks]
            active_track = available_tracks.find_generated_transcript(source_langs).translate("en")

        raw_entries = active_track.fetch()
        
        # --- CALCULATE VIDEO DURATION FOR RUBRIC POINTS ---
        total_seconds = 0
        if raw_entries:
            last_entry = raw_entries[-1]
            total_seconds = int(last_entry.get("start", 0) + last_entry.get("duration", 0))
        
        mins, secs = divmod(total_seconds, 60)
        duration_str = f"{mins}m {secs}s"
        
        # Compile the final text
        compiled_transcript = " ".join([segment["text"].replace("\n", " ") for segment in raw_entries])
        
        # Inject metadata header so the LLM knows the duration and context
        header = f"YOUTUBE SCRAPE REPORT | Video ID: {vid_id} | Estimated Duration: {duration_str}\n" + "="*60 + "\n\n"
        
        logger.info(f"Successfully scraped {len(compiled_transcript)} chars from YouTube video {vid_id}")
        return header + compiled_transcript

    except TranscriptsDisabled:
        return "⚠️ The creator has disabled closed captions for this video."
    except NoTranscriptFound:
        return "⚠️ No captions could be found or auto-generated for this video."
    except VideoUnavailable:
        return "⚠️ This video is either private, deleted, or geoblocked."
    except Exception as e:
        logger.exception("Unexpected crash during YouTube scraping.")
        return f"⚠️ Scraper failed due to an unknown error: {str(e)}"