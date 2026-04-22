from langchain_core.tools import tool
from utils.gmail_utils import fetch_recent_emails as base_fetch
from utils.calendar_utils import create_calendar_event

def format_emails(emails_list, include_date=True):
    """Utility to format email dictionaries into clean text."""
    if not emails_list:
        return "No emails found."
    lines = []
    for i, e in enumerate(emails_list, 1):
        lines.append(f"--- Email {i} ---")
        if include_date and "date" in e:
            lines.append(f"Date: {e['date']}")
        lines.append(f"From: {e['from']}")
        lines.append(f"Subject: {e['subject']}")
        lines.append(f"Snippet: {e.get('snippet', '')}")
    return "\n".join(lines)

@tool
def fetch_emails(limit: int = 5) -> str:
    """Fetch recent emails from the user's Gmail inbox.
    
    Args:
        limit (int): The number of recent emails to fetch (default 5).
        
    Returns:
        str: A formatted string containing the recent emails.
    """
    emails = base_fetch(limit=limit)
    return format_emails(emails)

@tool
def schedule_meeting(time: str, attendees: list[str], title: str = "AI Scheduled Meeting") -> str:
    """Schedule a meeting on Google Calendar and generate a Meet link.
    
    Args:
        time (str): The meeting time in 'YYYY-MM-DD HH:MM' format.
        attendees (list[str]): A list of email addresses for the attendees.
        title (str): The title of the meeting.
        
    Returns:
        str: A confirmation string with the Google Meet link.
    """
    # Handle string instead of list if LLM hallucinates
    if isinstance(attendees, str):
        attendees = [a.strip() for a in attendees.split(",") if a.strip()]
        
    link = create_calendar_event(time, attendees, title)
    return f"Meeting scheduled successfully! Meet Link: {link}"

@tool
def draft_email_reply(task_description: str, draft_content: str) -> str:
    """Draft a reply to an email based on a task description.
    
    Args:
        task_description (str): What the draft is supposed to accomplish.
        draft_content (str): The actual drafted email text.
        
    Returns:
        str: A confirmation that the draft was created.
    """
    # In a real app, this might save to a database or draft folder
    return f"Draft successfully created for task: '{task_description}'\n\nContent:\n{draft_content}"
