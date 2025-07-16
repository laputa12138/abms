import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

# Define task type constants for clarity and to avoid typos
TASK_TYPE_ANALYZE_TOPIC = "analyze_topic"
TASK_TYPE_GENERATE_OUTLINE = "generate_outline"
TASK_TYPE_PROCESS_CHAPTER = "process_chapter" # This might be a meta-task
TASK_TYPE_RETRIEVE_FOR_CHAPTER = "retrieve_for_chapter"
TASK_TYPE_WRITE_CHAPTER = "write_chapter"
TASK_TYPE_EVALUATE_CHAPTER = "evaluate_chapter"
TASK_TYPE_REFINE_CHAPTER = "refine_chapter"
TASK_TYPE_COMPILE_REPORT = "compile_report"
TASK_TYPE_GLOBAL_RETRIEVE_FOR_OUTLINE = "global_retrieve_for_outline"
TASK_TYPE_SUGGEST_OUTLINE_REFINEMENT = "suggest_outline_refinement"
TASK_TYPE_APPLY_OUTLINE_REFINEMENT = "apply_outline_refinement"
TASK_TYPE_RESOLVE_MISSING_CONTENT = "resolve_missing_content" # New Task Type

# Define chapter status constants
STATUS_PENDING = "pending"
STATUS_RETRIEVAL_NEEDED = "retrieval_needed"
STATUS_WRITING_NEEDED = "writing_needed"
STATUS_EVALUATION_NEEDED = "evaluation_needed"
STATUS_REFINEMENT_NEEDED = "refinement_needed"
STATUS_COMPLETED = "completed"
STATUS_ERROR = "error"


class WorkflowState:
    """
    Manages the dynamic state of the report generation workflow.
    Acts as a "working memory" or "central nervous system" for the agents and pipeline.
    """
    def __init__(self, user_topic: str, report_title: Optional[str] = None):
        self.workflow_id: str = str(uuid.uuid4())
        self.start_time: datetime = datetime.now()

        self.user_topic: str = user_topic
        self.report_title: Optional[str] = report_title or f"关于“{user_topic}”的分析报告"
        self.report_global_theme: Optional[str] = None # Added for global theme
        self.key_terms_definitions: Optional[Dict[str, str]] = None # Added for key terms

        self.topic_analysis_results: Optional[Dict[str, Any]] = None
        self.current_outline_md: Optional[str] = None
        # Parsed outline: List of {'title': str, 'level': int, 'id': str (unique key for chapter_data)}
        self.parsed_outline: List[Dict[str, Any]] = []

        # Chapter data: key is a unique chapter_id (e.g., from parsed_outline)
        self.chapter_data: Dict[str, Dict[str, Any]] = {}

        # Stores documents retrieved globally for each chapter ID before detailed chapter processing
        self.global_retrieved_docs_map: Optional[Dict[str, List[Dict[str, Any]]]] = None

        # Stores a single list of documents retrieved during the initial global topic analysis phase
        self.global_retrieved_docs: List[Dict[str, Any]] = []

        # Optional: A pool for caching retrieved information to avoid redundant searches
        # Key could be a normalized query string or a content hash.
        self.retrieved_information_pool: Dict[str, List[Dict[str, Any]]] = {}

        # Task queue: List of {'id': str, 'type': str, 'priority': int, 'payload': Dict, 'status': 'pending'/'processing'}
        self.pending_tasks: List[Dict[str, Any]] = []
        self.completed_tasks: List[Dict[str, Any]] = [] # For logging/auditing

        self.workflow_log: List[Tuple[datetime, str, Dict[str, Any]]] = []
        self.global_flags: Dict[str, Any] = {
            'data_loaded': False, # Becomes true after _process_and_load_data
            'topic_analyzed': False,
            'outline_generated': False,
            'outline_finalized': False, # Can be set to True to prevent further outline changes
            'report_compilation_requested': False, # Flag to trigger compilation when all else is done
            'report_generation_complete': False,
            'max_iterations_reached_for_all_chapters': False, # Placeholder
            'missing_content_resolution_requested': False, # New flag
            'missing_content_resolution_completed': False, # New flag
        }
        self.error_count: int = 0
        self.current_processing_task_id: Optional[str] = None

        self.log_event("WorkflowState initialized.", {"user_topic": user_topic, "report_title": self.report_title, "workflow_id": self.workflow_id})

    def log_event(self, message: str, details: Optional[Dict[str, Any]] = None, level: str = "INFO"): # Added level
        timestamp = datetime.now()
        # Store log level in details if not already there, for richer file logs
        log_details = details or {}
        if 'level' not in log_details: # Allow details to override default level for this entry
            log_details['level_implicit'] = level.upper()


        log_entry = (timestamp, message, log_details) # Storing tuple in memory
        self.workflow_log.append(log_entry)

        # For direct logger output, use the level
        if level.upper() == "ERROR":
            logger.error(f"[WF Log] {message} {log_details if log_details else ''}")
        elif level.upper() == "WARNING":
            logger.warning(f"[WF Log] {message} {log_details if log_details else ''}")
        elif level.upper() == "DEBUG":
            logger.debug(f"[WF Log] {message} {log_details if log_details else ''}")
        else: # INFO and others
            logger.info(f"[WF Log] {message} {log_details if log_details else ''}")


    def add_task(self, task_type: str, payload: Optional[Dict[str, Any]] = None, priority: int = 0) -> str:
        task_id = str(uuid.uuid4())
        task = {
            'id': task_id,
            'type': task_type,
            'priority': priority, # Lower number = higher priority
            'payload': payload or {},
            'status': 'pending', # 'pending', 'in_progress', 'completed', 'failed'
            'added_at': datetime.now()
        }
        self.pending_tasks.append(task)
        # Sort by priority (lower number first), then by time added (earlier first)
        self.pending_tasks.sort(key=lambda t: (t['priority'], t['added_at']))
        self.log_event(f"Task added: {task_type}", {"task_id": task_id, "priority": priority, "payload": payload})
        return task_id

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        if not self.pending_tasks:
            return None
        # Simple FIFO for tasks of the same priority (already sorted by priority then time)
        task = self.pending_tasks.pop(0)
        task['status'] = 'in_progress'
        self.current_processing_task_id = task['id']
        self.log_event(f"Task started: {task['type']}", {"task_id": task['id'], "payload": task['payload']})
        return task

    def complete_task(self, task_id: str, result_summary: Optional[str] = None, status: str = 'success'):
        if self.current_processing_task_id == task_id:
            self.current_processing_task_id = None

        # Find the task (it should have been moved from pending or handled if it was already processed)
        # For simplicity, we assume it was the one popped by get_next_task or handled if this is called directly
        # A more robust system might search for it in an 'in_progress_tasks' list.
        # For now, we just log its completion and move it to completed_tasks.

        # Find the original task dict to move it (this part is a bit simplified)
        # original_task_ref = None # Not used currently
        # This is inefficient, a better way would be to hold the task object.
        # For now, let's assume the calling context has the task dict.
        # We'll just record its completion.

        # Attempt to find more details about the task being completed if it was the current one
        task_type_for_log = "UnknownType"
        # This reconstruction is imperfect because task object is not passed to complete_task.
        # Agents should log their own completion with type. Orchestrator handles tasks it directly manages.
        # For now, rely on task_id.

        completed_task_info = {
            'id': task_id,
            'type': task_type_for_log, # Placeholder, ideally agent sets this or Orchestrator reconstructs
            'completed_at': datetime.now().isoformat(),
            'status': status,
            'message': result_summary or "N/A" # Using message field consistent with add_task
        }
        self.completed_tasks.append(completed_task_info)

        log_level = "INFO"
        if status == 'failed':
            self.increment_error_count()
            log_level = "ERROR"

        self.log_event(f"Task completed: {task_id} (Type: {task_type_for_log})",
                       {"status": status, "result_summary": result_summary}, level=log_level)

        if self.current_processing_task_id == task_id:
            self.current_processing_task_id = None # Clear if it was the active task

        # Specific post-completion actions based on task type
        # This requires knowing the task type. If an agent calls this, it should ensure
        # workflow_state is updated with any flags. Orchestrator handles its own managed tasks' flags.
        # Example for a task type that this method might know about (e.g. if Orchestrator calls it):
        # if task_type_for_log == TASK_TYPE_RESOLVE_MISSING_CONTENT and status == 'success':
        #    self.set_flag('missing_content_resolution_completed', True)
        # This is now handled more directly by agents or orchestrator setting flags.
        # However, if an Agent calls complete_task, it's also responsible for setting its output flags.
        # Let's add the specific flag setting for RESOLVE_MISSING_CONTENT here if it's completed.
        # This assumes 'task_type_for_log' could be correctly identified or passed.
        # For a cleaner design, agent's execute_task should set this flag before calling complete_task.
        # Let's assume for now that the MissingContentResolutionAgent will set this flag itself.
        # The Orchestrator will set it for tasks it manages directly.
        # The logic in complete_task in previous iteration was:
        # if task_type_of_completed_task == TASK_TYPE_RESOLVE_MISSING_CONTENT and status == 'success':
        #    self.set_flag('missing_content_resolution_completed', True)
        # This requires task_type_of_completed_task to be known.
        # The MissingContentResolutionAgent itself calls complete_task and will set this flag.


    def update_topic_analysis(self, results: Dict[str, Any]):
        self.topic_analysis_results = results
        self.set_flag('topic_analyzed', True)
        self.log_event("Topic analysis results updated.", results)

    def update_outline(self, outline_md: str, parsed_outline: List[Dict[str, Any]]):
        self.current_outline_md = outline_md
        self.parsed_outline = [] # Reset before populating

        # Initialize chapter_data based on the new parsed outline
        # Ensure each outline item has a unique ID for chapter_data key
        temp_chapter_data = {}
        for i, item in enumerate(parsed_outline):
            # item should be like {'title': str, 'level': int} from ReportCompilerAgent._parse_markdown_outline
            # We need a persistent key. Using title might be problematic if titles change.
            # Let's generate a simple unique key/ID for each outline item.
            chapter_key = item.get('id', f"chapter_{i}_{str(uuid.uuid4())[:4]}") # Use existing ID or generate
            item['id'] = chapter_key # Ensure 'id' exists in parsed_outline item

            self.parsed_outline.append(item) # Add item with ID to state's parsed_outline

            # If chapter_key already exists, try to preserve some data, otherwise initialize
            existing_data = self.chapter_data.get(chapter_key, {})
            temp_chapter_data[chapter_key] = {
                'title': item['title'], # Keep title in chapter_data for convenience
                'level': item['level'],
                'status': existing_data.get('status', STATUS_PENDING), # Preserve status if exists
                'content': existing_data.get('content'),
                'retrieved_docs': existing_data.get('retrieved_docs'),
                'evaluations': existing_data.get('evaluations', []),
                'versions': existing_data.get('versions', []),
                'errors': existing_data.get('errors', []),
                'citations_structured': existing_data.get('citations_structured', []) # New field for structured citations
            }
        self.chapter_data = temp_chapter_data # Replace with new structure
        self.set_flag('outline_generated', True)
        self.set_flag('outline_finalized', False) # New outline means it's not finalized yet
        self.log_event("Outline updated.", {"outline_md_preview": outline_md[:100]+"...", "num_chapters": len(self.parsed_outline)})

    def _get_chapter_entry(self, chapter_key: str, create_if_missing: bool = False) -> Optional[Dict[str, Any]]:
        """Helper to get or optionally create a chapter_data entry."""
        if chapter_key not in self.chapter_data:
            if create_if_missing:
                # Try to find title/level from parsed_outline if key matches an ID
                outline_item = next((item for item in self.parsed_outline if item.get('id') == chapter_key), None)
                title = outline_item.get('title', chapter_key) if outline_item else chapter_key
                level = outline_item.get('level', 0) if outline_item else 0

                self.chapter_data[chapter_key] = {
                    'title': title, 'level': level, 'status': STATUS_PENDING,
                    'content': None, 'retrieved_docs': None,
                    'evaluations': [], 'versions': [], 'errors': [],
                    'citations_structured': [] # Initialize new field
                }
                self.log_event(f"Chapter entry created on demand: {chapter_key}")
            else:
                logger.warning(f"Accessing non-existent chapter_key '{chapter_key}' without create_if_missing.")
                return None
        return self.chapter_data[chapter_key]

    def update_chapter_status(self, chapter_key: str, status: str):
        entry = self._get_chapter_entry(chapter_key, create_if_missing=True)
        if entry:
            entry['status'] = status
            self.log_event(f"Chapter '{chapter_key}' status updated to: {status}")

    def update_chapter_content(self, chapter_key: str, content: str,
                               retrieved_docs: Optional[List[Dict[str, Any]]] = None,
                               citations_structured_list: Optional[List[Dict[str, Any]]] = None, # New parameter
                               is_new_version: bool = True):
        entry = self._get_chapter_entry(chapter_key, create_if_missing=True)
        if entry:
            if is_new_version and entry.get('content'): # Save previous version
                entry['versions'].append(entry['content'])
            entry['content'] = content
            if retrieved_docs is not None: # Update if new docs are provided
                entry['retrieved_docs'] = retrieved_docs
            if citations_structured_list is not None: # Store structured citations
                entry['citations_structured'] = citations_structured_list
            self.log_event(f"Chapter '{chapter_key}' content updated.", {
                "content_length": len(content),
                "num_structured_citations": len(citations_structured_list) if citations_structured_list else 0
            })

    def add_chapter_evaluation(self, chapter_key: str, evaluation: Dict[str, Any]):
        entry = self._get_chapter_entry(chapter_key, create_if_missing=True)
        if entry:
            entry['evaluations'].append(evaluation)
            self.log_event(f"Evaluation added for chapter '{chapter_key}'.", {"score": evaluation.get('score')})

    def add_chapter_error(self, chapter_key: str, error_message: str):
        entry = self._get_chapter_entry(chapter_key, create_if_missing=True)
        if entry:
            entry['errors'].append(f"[{datetime.now().isoformat()}] {error_message}")
            entry['status'] = STATUS_ERROR
            self.log_event(f"Error recorded for chapter '{chapter_key}'.", {"error": error_message})


    def set_flag(self, flag_name: str, value: Any):
        self.global_flags[flag_name] = value
        self.log_event(f"Global flag '{flag_name}' set to: {value}")

    def get_flag(self, flag_name: str, default: Optional[Any] = None) -> Any:
        return self.global_flags.get(flag_name, default)

    def get_chapter_data(self, chapter_key: str) -> Optional[Dict[str, Any]]:
        return self.chapter_data.get(chapter_key)

    def get_all_chapter_keys_by_status(self, status: str) -> List[str]:
        return [key for key, data in self.chapter_data.items() if data.get('status') == status]

    def are_all_chapters_completed(self) -> bool:
        if not self.parsed_outline: return False # No outline means nothing to complete
        for item in self.parsed_outline:
            chapter_key = item['id']
            chapter_info = self.chapter_data.get(chapter_key)
            if not chapter_info or chapter_info.get('status') != STATUS_COMPLETED:
                return False
        return True

    def count_completed_chapters(self) -> int:
        """Counts how many chapters (from parsed_outline) are marked as COMPLETED."""
        if not self.parsed_outline:
            return 0
        count = 0
        for item in self.parsed_outline:
            chapter_key = item['id']
            chapter_info = self.chapter_data.get(chapter_key)
            if chapter_info and chapter_info.get('status') == STATUS_COMPLETED:
                count += 1
        return count

    def are_all_chapter_tasks_processed_or_terminal(self) -> bool:
        """
        Checks if all chapter-specific tasks are done or if chapters are in a terminal state
        (e.g., completed or errored out with no more processing tasks for them).
        This is used by Orchestrator to decide when to move to MissingContentResolution or Compilation.
        """
        if not self.parsed_outline or not self.get_flag('outline_finalized', False):
            # If there's no outline or it's not finalized, chapter processing hasn't meaningfully concluded.
            self.log_event("are_all_chapter_tasks_processed_or_terminal: False (no parsed_outline or outline not finalized).", level="DEBUG")
            return False

        # Check if any chapter-specific tasks are still pending in the main task queue.
        # These are tasks that would lead to content generation or modification for a chapter.
        chapter_processing_task_types = [
            TASK_TYPE_PROCESS_CHAPTER, TASK_TYPE_RETRIEVE_FOR_CHAPTER,
            TASK_TYPE_WRITE_CHAPTER, TASK_TYPE_EVALUATE_CHAPTER, TASK_TYPE_REFINE_CHAPTER
        ]
        for task in self.pending_tasks:
            if task['type'] in chapter_processing_task_types:
                self.log_event(f"are_all_chapter_tasks_processed_or_terminal: False (pending task found: {task['id']}, type: {task['type']}).", level="DEBUG")
                return False # Found an active chapter processing task, so not all are done.

        # If the queue is clear of these tasks, it implies that for every chapter,
        # either it reached STATUS_COMPLETED, or it reached STATUS_ERROR and no further
        # automated processing tasks (like refine/rewrite loops) were added for it.
        # The MissingContentResolutionAgent will then get a chance to inspect actual content.
        self.log_event("are_all_chapter_tasks_processed_or_terminal: True (no pending chapter-specific tasks in queue).", level="DEBUG")
        return True


    def increment_error_count(self):
        self.error_count += 1
        self.log_event("Global error count incremented.", {"current_error_count": self.error_count})

    def get_full_report_context_for_compilation(self) -> Dict[str, Any]:
        """Prepares data needed by ReportCompilerAgent."""
        # Filter chapter_data to only include chapters present in the current parsed_outline
        # and format it as expected by ReportCompilerAgent (title -> content string)
        valid_chapter_contents = {}
        for item in self.parsed_outline:
            chapter_key = item['id']
            data = self.chapter_data.get(chapter_key)
            if data and data.get('status') == STATUS_COMPLETED and data.get('content'):
                valid_chapter_contents[data['title']] = data['content'] # Use title as key for compiler
            else:
                 logger.warning(f"Chapter '{data.get('title', chapter_key) if data else chapter_key}' not completed or has no content for compilation.")


        return {
            "report_title": self.report_title,
            "markdown_outline": self.current_outline_md,
            "chapter_contents": valid_chapter_contents, # Dict[chapter_title, chapter_text_content]
            "report_topic_details": self.topic_analysis_results
        }

    def apply_refinements_to_parsed_outline(
        self,
        current_parsed_outline: List[Dict[str, Any]],
        refinement_operations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Applies a list of refinement operations to a parsed outline.
        Returns a new parsed_outline list.
        Does not modify the state's own outline directly; Orchestrator will use the result.
        Operations should be validated before being passed here.
        """
        import copy # Ensure copy is imported
        new_parsed_outline = copy.deepcopy(current_parsed_outline)

        # Helper to find an item's index by ID
        def find_item_index(outline_list, item_id):
            if item_id is None: return -1 # Cannot find a None item_id
            for idx, item in enumerate(outline_list):
                if item.get('id') == item_id:
                    return idx
            return -1

        for op in refinement_operations:
            action = op['action']
            item_id = op.get('id') # Used by most actions

            if action == "add":
                new_item = {
                    "id": f"ch_{str(uuid.uuid4())[:8]}",
                    "title": op['title'],
                    "level": op['level'],
                }
                after_id = op.get('after_id')
                insert_index = -1
                if after_id: # if after_id is provided, find its index
                    idx = find_item_index(new_parsed_outline, after_id)
                    if idx != -1:
                        insert_index = idx + 1
                    else: # after_id not found, append to end (or log warning)
                        logger.warning(f"apply_refinements: 'add' action's after_id '{after_id}' not found. Appending.")
                        new_parsed_outline.append(new_item)
                        continue

                if insert_index != -1: # Found after_id or it was None (implies append/prepend based on context)
                    new_parsed_outline.insert(insert_index, new_item)
                else: # after_id was None or not found, append to the end
                    new_parsed_outline.append(new_item)

            elif action == "delete":
                idx = find_item_index(new_parsed_outline, item_id)
                if idx != -1:
                    del new_parsed_outline[idx]
                else:
                    logger.warning(f"apply_refinements: 'delete' action's id '{item_id}' not found.")

            elif action == "modify_title":
                idx = find_item_index(new_parsed_outline, item_id)
                if idx != -1:
                    new_parsed_outline[idx]['title'] = op['new_title']
                else:
                    logger.warning(f"apply_refinements: 'modify_title' action's id '{item_id}' not found.")

            elif action == "modify_level":
                idx = find_item_index(new_parsed_outline, item_id)
                if idx != -1:
                    # Fix for the typo 'new:level' -> 'new_level'
                    if 'new:level' in op:
                        op['new_level'] = op.pop('new:level')
                    new_parsed_outline[idx]['level'] = op['new_level']
                else:
                    logger.warning(f"apply_refinements: 'modify_level' action's id '{item_id}' not found.")

            elif action == "move":
                item_idx = find_item_index(new_parsed_outline, item_id)
                if item_idx == -1:
                    logger.warning(f"apply_refinements: 'move' action's item_id '{item_id}' not found.")
                    continue

                item_to_move = new_parsed_outline.pop(item_idx)

                after_id = op.get('after_id') # ID of chapter to move it after
                # before_id = op.get('before_id') # ID of chapter to move it before (optional alternative)

                insert_at_idx = -1 # Default to indicating not found or prepend

                if after_id is not None:
                    target_idx = find_item_index(new_parsed_outline, after_id)
                    if target_idx != -1:
                        insert_at_idx = target_idx + 1
                    else: # after_id specified but not found, could append or error
                        logger.warning(f"apply_refinements: 'move' action's after_id '{after_id}' not found. Appending item '{item_id}'.")
                        new_parsed_outline.append(item_to_move)
                        continue
                # elif before_id is not None:
                #     target_idx = find_item_index(new_parsed_outline, before_id)
                #     if target_idx != -1:
                #         insert_at_idx = target_idx
                #     else: # before_id specified but not found
                #          logger.warning(f"apply_refinements: 'move' action's before_id '{before_id}' not found. Prepending item '{item_id}'.")
                #          new_parsed_outline.insert(0, item_to_move)
                #          continue

                if insert_at_idx != -1 : # Specific index found via after_id (or potentially before_id)
                     new_parsed_outline.insert(insert_at_idx, item_to_move)
                elif after_id is None: # No after_id (and no before_id implies move to start)
                    new_parsed_outline.insert(0, item_to_move) # Default: move to beginning if no specific target
                # else: after_id was not None, but not found, already handled by appending.

            elif action == "merge":
                primary_idx = find_item_index(new_parsed_outline, op['primary_id'])
                secondary_idx = find_item_index(new_parsed_outline, op['secondary_id'])

                if primary_idx != -1 and secondary_idx != -1:
                    if primary_idx == secondary_idx:
                        logger.warning(f"apply_refinements: 'merge' action's primary and secondary IDs are the same ('{op['primary_id']}'). Skipping.")
                        continue
                    if 'new_title_for_primary' in op:
                        new_parsed_outline[primary_idx]['title'] = op['new_title_for_primary']

                    # Ensure secondary_idx is still valid after potential list modifications if primary_idx < secondary_idx
                    # However, standard list deletion handles index shifts correctly if deleting higher index first, or if they are distinct.
                    # For safety, if primary_idx was before secondary_idx and an element was removed, secondary_idx might shift.
                    # But since we are finding them independently and then deleting one, it's safer.
                    # If we pop secondary_idx, primary_idx remains valid if it was smaller.
                    # If primary_idx was larger, its index would shift if secondary_idx was popped first.
                    # Simplest: delete by value or re-find index if necessary, but direct index deletion is fine if careful.
                    del new_parsed_outline[secondary_idx]
                else:
                    logger.warning(f"apply_refinements: 'merge' action's primary_id '{op.get('primary_id')}' or secondary_id '{op.get('secondary_id')}' not found.")

            elif action == "split":
                item_idx = find_item_index(new_parsed_outline, item_id)
                if item_idx != -1:
                    original_item_level = new_parsed_outline[item_idx].get('level', 1) # Get level of item being split
                    del new_parsed_outline[item_idx] # Remove original

                    for i, new_chap_info in enumerate(op['new_chapters']):
                        new_item_level = new_chap_info.get('level', original_item_level) # Use new level or inherit
                        new_item = {
                            "id": f"ch_{str(uuid.uuid4())[:8]}",
                            "title": new_chap_info['title'],
                            "level": new_item_level,
                        }
                        new_parsed_outline.insert(item_idx + i, new_item)
                else:
                    logger.warning(f"apply_refinements: 'split' action's id '{item_id}' not found.")

        return new_parsed_outline

    def generate_markdown_from_parsed_outline(self, parsed_outline: List[Dict[str, Any]]) -> str:
        """
        Generates a Markdown string representation from a parsed outline structure.
        Assumes 'title' and 'level' keys in each dictionary.
        Level 1 -> # Title
        Level 2 -> ## Title
        Level 3+ -> uses '-' with indentation relative to level 2.
                     e.g. Level 3 is "- Title", Level 4 is "  - Title"
        """
        md_lines = []
        for item in parsed_outline:
            title = item.get('title', 'Untitled')
            level = item.get('level', 1)
            if level == 1:
                md_lines.append(f"# {title}")
            elif level == 2:
                md_lines.append(f"## {title}")
            else: # level 3+
                # Indentation: level 3 has 0 spaces, level 4 has 2, level 5 has 4, etc.
                indent_count = (level - 3) * 2
                indent = " " * indent_count
                md_lines.append(f"{indent}- {title}")
        return "\n".join(md_lines)

    def set_global_retrieved_docs_map(self, docs_map: Dict[str, List[Dict[str, Any]]]):
        """Sets the map of globally retrieved documents per chapter."""
        self.global_retrieved_docs_map = docs_map
        self.log_event("Global retrieved documents map (per-chapter) updated.", {"num_chapters_with_docs": len(docs_map)})

    def get_global_retrieved_docs_map(self) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """Gets the map of globally retrieved documents per chapter."""
        return self.global_retrieved_docs_map

    def set_global_retrieved_docs(self, docs: List[Dict[str, Any]]):
        """Sets the list of globally retrieved documents from the initial analysis phase."""
        self.global_retrieved_docs = docs
        self.log_event("Global retrieved documents list updated.", {"doc_count": len(docs)})

    def get_global_retrieved_docs(self) -> List[Dict[str, Any]]:
        """Gets the list of globally retrieved documents."""
        return self.global_retrieved_docs

    def update_report_global_theme(self, theme: str):
        """Updates the global theme/core idea of the report."""
        self.report_global_theme = theme
        self.log_event("Report global theme updated.", {"theme_preview": theme[:100]+"..."})

    def get_report_global_theme(self) -> Optional[str]:
        """Gets the global theme/core idea of the report."""
        return self.report_global_theme

    def update_key_terms_definitions(self, definitions: Dict[str, str]):
        """Updates the definitions for key terms."""
        self.key_terms_definitions = definitions
        self.log_event("Key terms definitions updated.", {"terms_count": len(definitions)})

    def get_key_terms_definitions(self) -> Optional[Dict[str, str]]:
        """Gets the definitions for key terms."""
        return self.key_terms_definitions


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger.info("WorkflowState Example Start")

    state = WorkflowState(user_topic="AI in Education", report_title="The Impact of AI on Education")

    # Test new methods for global theme and key terms
    state.update_report_global_theme("This report explores the multifaceted impact of Artificial Intelligence on modern educational paradigms, focusing on personalized learning, administrative efficiency, and ethical considerations.")
    state.update_key_terms_definitions({
        "AI": "Artificial Intelligence - The theory and development of computer systems able to perform tasks normally requiring human intelligence.",
        "ML": "Machine Learning - A subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed."
    })
    print(f"\nReport Global Theme: {state.get_report_global_theme()}")
    print(f"Key Terms Definitions: {json.dumps(state.get_key_terms_definitions(), indent=2, ensure_ascii=False)}")


    # Add initial task
    state.add_task(TASK_TYPE_ANALYZE_TOPIC, payload={"user_topic": state.user_topic}, priority=1)

    next_task = state.get_next_task()
    print(f"\nNext task to process: {json.dumps(next_task, indent=2, default=str)}")

    # Simulate completing the task
    if next_task:
        state.update_topic_analysis({"generalized_topic_cn": "人工智能教育", "keywords_cn": ["AI", "教育"]})
        state.complete_task(next_task['id'], result_summary="Topic analyzed successfully.")
        state.add_task(TASK_TYPE_GENERATE_OUTLINE, payload={"topic_details": state.topic_analysis_results}, priority=2)

    next_task = state.get_next_task()
    print(f"\nNext task to process: {json.dumps(next_task, indent=2, default=str)}")

    # Simulate outline generation
    if next_task and next_task['type'] == TASK_TYPE_GENERATE_OUTLINE:
        mock_outline_md = "- Introduction\n  - Background\n- Main Body\n- Conclusion"
        mock_parsed_outline = [
            {'title': 'Introduction', 'level': 1, 'id': 'chap_intro'},
            {'title': 'Background', 'level': 2, 'id': 'chap_intro_bg'},
            {'title': 'Main Body', 'level': 1, 'id': 'chap_main'},
            {'title': 'Conclusion', 'level': 1, 'id': 'chap_conc'}
        ]
        state.update_outline(mock_outline_md, mock_parsed_outline)
        state.complete_task(next_task['id'], result_summary="Outline generated.")

        # Add tasks for each chapter based on new outline structure
        for item in state.parsed_outline:
            state.add_task(TASK_TYPE_PROCESS_CHAPTER, payload={'chapter_key': item['id'], 'chapter_title': item['title']}, priority=3)

    print(f"\nPending tasks after outline generation: {len(state.pending_tasks)}")
    for task in state.pending_tasks:
        print(task)

    # Simulate processing one chapter
    chapter_task = state.get_next_task() # Should be process_chapter for 'chap_intro'
    if chapter_task and chapter_task['type'] == TASK_TYPE_PROCESS_CHAPTER:
        chap_key = chapter_task['payload']['chapter_key']
        state.update_chapter_status(chap_key, STATUS_RETRIEVAL_NEEDED)
        # Simulate retrieval
        state.chapter_data[chap_key]['retrieved_docs'] = [{"document": "Some retrieved parent context for intro.", "score": 0.9}]
        state.update_chapter_status(chap_key, STATUS_WRITING_NEEDED)
        # Simulate writing
        state.update_chapter_content(chap_key, "This is the written introduction.", retrieved_docs=state.chapter_data[chap_key]['retrieved_docs'])
        state.update_chapter_status(chap_key, STATUS_EVALUATION_NEEDED)
        # Simulate evaluation
        state.add_chapter_evaluation(chap_key, {"score": 70, "feedback_cn": "Needs more detail."})
        state.update_chapter_status(chap_key, STATUS_REFINEMENT_NEEDED)
        # Simulate refinement
        state.update_chapter_content(chap_key, "This is the refined and more detailed introduction.", is_new_version=True)
        state.update_chapter_status(chap_key, STATUS_COMPLETED) # Assume refinement was good enough
        state.complete_task(chapter_task['id'], result_summary=f"Chapter {chap_key} processed.")

    print(f"\nChapter data for '{chapter_task['payload']['chapter_key'] if chapter_task else ''}':")
    if chapter_task : print(json.dumps(state.get_chapter_data(chapter_task['payload']['chapter_key']), indent=2, default=str))

    # Test global retrieved docs map
    mock_global_docs = {
        "chap_intro": [{"title": "Global Doc A", "text": "Content for intro"}],
        "chap_main": [{"title": "Global Doc B", "text": "Content for main"}]
    }
    state.set_global_retrieved_docs_map(mock_global_docs)
    retrieved_map = state.get_global_retrieved_docs_map()
    print(f"\nGlobal retrieved docs map set and get: {'Success' if retrieved_map == mock_global_docs else 'Failure'}")
    assert retrieved_map == mock_global_docs

    print(f"\nAre all chapters completed? {state.are_all_chapters_completed()}") # Will be false

    # Simulate completing all other chapters
    while True:
        task = state.get_next_task()
        if not task: break
        if task['type'] == TASK_TYPE_PROCESS_CHAPTER:
            key = task['payload']['chapter_key']
            state.update_chapter_content(key, f"Content for {state.chapter_data[key]['title']}.")
            state.update_chapter_status(key, STATUS_COMPLETED)
            state.complete_task(task['id'])

    print(f"\nAre all chapters completed after mock processing? {state.are_all_chapters_completed()}") # Should be true

    if state.are_all_chapters_completed():
        state.set_flag('outline_finalized', True) # Assuming outline is now final
        state.add_task(TASK_TYPE_COMPILE_REPORT, priority=100) # Low priority, run at the end

    compile_task = state.get_next_task()
    if compile_task and compile_task['type'] == TASK_TYPE_COMPILE_REPORT:
        report_context = state.get_full_report_context_for_compilation()
        print("\nContext for report compilation:")
        print(json.dumps(report_context, indent=2, default=str))
        state.complete_task(compile_task['id'])
        state.set_flag('report_generation_complete', True)

    print(f"\nWorkflow log entries: {len(state.workflow_log)}")
    # print("Last log entry:", state.workflow_log[-1] if state.workflow_log else "None")

    logger.info("WorkflowState Example End")
