"""
Smart Ollama Manager for Generation Two
Intelligently manages Ollama connections with fallback and error handling
"""

# Use print for immediate visibility during import
print("[ollama_manager] Starting imports...", flush=True)

import logging
print("[ollama_manager]   ✓ logging", flush=True)
import time
import threading
print("[ollama_manager]   ✓ time, threading", flush=True)
from typing import Optional, List, Dict, Callable
from datetime import datetime, timedelta
print("[ollama_manager]   ✓ typing, datetime", flush=True)

# V2 style: Import ollama LAZILY (only when needed) to avoid blocking during module import
# CRITICAL: Do NOT import ollama at module level - it may try to connect to server and block
print("[ollama_manager] Setting up imports (ollama will be lazy)...", flush=True)

# Always set up requests fallback (this is safe and fast)
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
print("[ollama_manager]   ✓ requests imported", flush=True)
print("[ollama_manager]   ℹ ollama will be imported lazily when first used", flush=True)

# Import modularized utilities
from .ollama_import import get_ollama_chat_function, get_ollama_list_function, import_ollama_library
from .ollama_health import (
    get_model_names_from_ollama,
    get_model_names_from_requests,
    find_best_model_match,
    select_alternative_model
)
from .ollama_request import (
    call_ollama_library,
    call_ollama_requests,
    create_progress_monitor
)
print("[ollama_manager]   ✓ modularized utilities imported", flush=True)

# OLLAMA_AVAILABLE is kept for backward compatibility but always False at module level
# We'll do lazy imports when actually needed (in methods, not at module level)
OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)
print("[ollama_manager] All imports completed", flush=True)


class OllamaManager:
    """
    Smart Ollama manager with connection pooling, fallback, and rate limiting
    
    Features:
    - Connection health monitoring
    - Automatic fallback to alternative methods
    - Rate limiting to prevent overload
    - Connection pooling
    - Smart retry logic
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:1.5b",
        timeout: int = 120,
        max_retries: int = 3,
        rate_limit: float = 2.0  # seconds between requests
    ):
        """
        Initialize Ollama manager
        
        Args:
            base_url: Ollama server URL
            model: Model name to use
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            rate_limit: Minimum seconds between requests
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        
        # Connection state
        self.is_available = False
        self.last_check = None
        self.last_request_time = 0.0
        self.health_check_interval = 300  # Check health every 5 minutes
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'fallback_used': 0
        }
        
        # V2 style: Use ollama library directly (no session needed if available)
        if OLLAMA_AVAILABLE:
            logger.debug("Using ollama library (V2 style - no session needed)")
            self.session = None  # Not needed when using ollama library
        else:
            # Fallback: Create session with connection pooling for requests
            try:
                self.session = requests.Session()
                retry_strategy = Retry(
                    total=1,  # Only retry once (we handle retries ourselves)
                    backoff_factor=0.1,
                    status_forcelist=[429, 500, 502, 503, 504],
                )
                adapter = HTTPAdapter(
                    max_retries=retry_strategy,
                    pool_connections=10,  # Allow up to 10 concurrent connections
                    pool_maxsize=10,  # Max connections per pool
                    pool_block=False  # Don't block if pool is full, raise exception instead
                )
                self.session.mount("http://", adapter)
                self.session.mount("https://", adapter)
                logger.debug("Session with connection pooling initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize session with connection pooling: {e}", exc_info=True)
                # Fallback to regular Session without pooling
                self.session = requests.Session()
                logger.warning("Using basic Session (no connection pooling)")
        
        # Check initial availability (defer to avoid blocking import)
        # Don't check during __init__ - let it happen lazily on first use
        # self._check_availability()  # Commented out to prevent blocking during import
    
    def _check_availability(self) -> bool:
        """Check if Ollama is available and model exists"""
        try:
            # Try ollama library first
            list_func = get_ollama_list_function()
            model_names = []
            
            if list_func:
                model_names = get_model_names_from_ollama(list_func)
            
            # Fallback to requests if ollama library not available
            if not model_names and self.session:
                model_names = get_model_names_from_requests(self.session, self.base_url)
            
            if not model_names:
                self.is_available = False
                self.consecutive_failures += 1
                return False
            
            # Find best model match
            requested_model = self.model
            model_exists, matched_model, base_matches = find_best_model_match(model_names, requested_model)
            
            if matched_model:
                self.model = matched_model
                if self.model != requested_model:
                    logger.info(f"Using available model: {self.model} (requested: {requested_model})")
                self.is_available = True
                self.consecutive_failures = 0
                self.last_check = datetime.now()
                logger.info(f"Ollama is available at {self.base_url} with model {self.model}")
                return True
            
            # Try to find an alternative model
            preferred = [self.model]
            if "qwen" in self.model.lower() or "coder" in self.model.lower():
                preferred = ["qwen2.5-coder:1.5b", "qwen2.5-coder:7b", "qwen2.5-coder:32b", self.model]
            
            alternative = self._find_available_model(preferred)
            if alternative:
                logger.info(f"Model '{self.model}' not found, using alternative: {alternative}")
                self.model = alternative
                self.is_available = True
                self.consecutive_failures = 0
                self.last_check = datetime.now()
                return True
            else:
                logger.warning(f"Model '{self.model}' not found in Ollama. Available models: {', '.join(model_names[:5])}")
                self.is_available = False
                return False
                
        except Exception as e:
            logger.debug(f"Ollama availability check failed: {e}")
        
        self.is_available = False
        self.consecutive_failures += 1
        return False
    
    def _find_available_model(self, preferred_models: List[str] = None) -> Optional[str]:
        """Find an available model from a list of preferred models"""
        try:
            # V2 style: Try ollama library first (lazy import), fallback to requests
            # IMPORTANT: Import directly from site-packages to bypass local generation_two/ollama module
            try:
                import importlib.util
                import sys
                
                # Temporarily remove local generation_two/ollama from modules
                original_ollama = sys.modules.pop('ollama', None)
                original_generation_two_ollama = sys.modules.pop('generation_two.ollama', None)
                
                try:
                    # More aggressive search: directly search site-packages directories
                    import site
                    import os
                    
                    # Get all site-packages directories
                    site_packages_dirs = site.getsitepackages()
                    if hasattr(site, 'getusersitepackages'):
                        user_site = site.getusersitepackages()
                        if user_site:
                            site_packages_dirs.append(user_site)
                    
                    # Try to find ollama in site-packages
                    ollama_pkg = None
                    list_func = None
                    for site_dir in site_packages_dirs:
                        ollama_path = os.path.join(site_dir, 'ollama')
                        if os.path.exists(ollama_path) or os.path.exists(ollama_path + '.py'):
                            try:
                                spec = importlib.util.spec_from_file_location(
                                    'ollama_installed',
                                    os.path.join(ollama_path, '__init__.py') if os.path.isdir(ollama_path) else ollama_path + '.py'
                                )
                                if spec and spec.loader:
                                    ollama_pkg = importlib.util.module_from_spec(spec)
                                    spec.loader.exec_module(ollama_pkg)
                                    list_func = getattr(ollama_pkg, 'list', None)
                                    if list_func:
                                        sys.modules['_ollama_pkg_installed'] = ollama_pkg
                                        models = list_func()
                                        if isinstance(models, dict):
                                            available_names = [m.get('name', '') for m in models.get('models', [])]
                                        else:
                                            available_names = [m.name if hasattr(m, 'name') else str(m) for m in models.models] if hasattr(models, 'models') else []
                                        break
                            except Exception as e:
                                logger.debug(f"Error loading ollama from {site_dir}: {e}")
                                continue
                    
                    # Fallback: try find_spec
                    if not list_func:
                        spec = importlib.util.find_spec('ollama')
                        if spec and spec.loader and spec.origin:
                            origin_lower = spec.origin.lower()
                            if 'site-packages' in origin_lower or 'dist-packages' in origin_lower:
                                ollama_pkg = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(ollama_pkg)
                                list_func = getattr(ollama_pkg, 'list', None)
                                if list_func:
                                    sys.modules['_ollama_pkg_installed'] = ollama_pkg
                                    models = list_func()
                                    if isinstance(models, dict):
                                        available_names = [m.get('name', '') for m in models.get('models', [])]
                                    else:
                                        available_names = [m.name if hasattr(m, 'name') else str(m) for m in models.models] if hasattr(models, 'models') else []
                                else:
                                    raise ImportError("ollama package found but no list function")
                            else:
                                raise ImportError(f"ollama spec not in site-packages (origin: {spec.origin})")
                        else:
                            raise ImportError(f"ollama spec not found")
                finally:
                    # Restore original modules if they existed
                    if original_ollama:
                        sys.modules['ollama'] = original_ollama
                    if original_generation_two_ollama:
                        sys.modules['generation_two.ollama'] = original_generation_two_ollama
            except (ImportError, Exception) as e:
                logger.debug(f"Ollama list() failed: {e}, using requests fallback")
                # Fall back to requests
                if not self.session:
                    return None
                available_names = get_model_names_from_requests(self.session, self.base_url)
            
            if available_names:
                # Find best match from preferred models
                return select_alternative_model(available_names, preferred_models or [])
        except Exception as e:
            logger.debug(f"Error finding available model: {e}")
        
        return None
    
    def _should_check_health(self) -> bool:
        """Determine if health check is needed"""
        if self.last_check is None:
            return True
        
        elapsed = (datetime.now() - self.last_check).total_seconds()
        return elapsed > self.health_check_interval
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting (per-thread to allow concurrency)"""
        # Use thread-local storage to allow concurrent requests from different threads
        # This prevents one thread from blocking others
        if not hasattr(self, '_thread_local'):
            self._thread_local = threading.local()
        
        current_time = time.time()
        thread_last_time = getattr(self._thread_local, 'last_request_time', 0.0)
        time_since_last = current_time - thread_last_time
        
        # Only enforce rate limit if this thread made a request recently
        # This allows multiple threads to make concurrent requests
        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            time.sleep(sleep_time)
        
        self._thread_local.last_request_time = time.time()
        # Also update global for stats (we're already in a lock, so just update directly)
        self.last_request_time = time.time()
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,  # Increased for better context handling
        progress_callback: Optional[callable] = None
    ) -> Optional[str]:
        """
        Generate text using Ollama
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text or None if failed
        """
        # V2 style: NO rate limiting - let threads run freely (ollama library handles concurrency)
        # Rate limiting was causing blocking - removed for true concurrency
        
        # Acquire lock only for shared state checks (not for the entire request)
        logger.debug(f"generate() called - entering lock, consecutive_failures: {self.consecutive_failures}, max: {self.max_consecutive_failures}")
        with self.lock:
            # Check health periodically - this will reset consecutive_failures if Ollama is available
            if self._should_check_health():
                logger.debug("Health check needed, checking availability...")
                if self._check_availability():
                    # If health check succeeds, reset consecutive failures
                    logger.info(f"Ollama health check passed, resetting consecutive_failures from {self.consecutive_failures} to 0")
                    self.consecutive_failures = 0
            
            # If too many failures, try a quick health check first before giving up
            if self.consecutive_failures >= self.max_consecutive_failures:
                logger.debug(f"Too many consecutive failures ({self.consecutive_failures}), doing quick health check...")
                # Quick check: try to import ollama and see if it's available
                # IMPORTANT: Import directly from site-packages to bypass local generation_two/ollama module
                try:
                    import importlib.util
                    import sys
                    
                    # Temporarily remove local generation_two/ollama from modules
                    original_ollama = sys.modules.pop('ollama', None)
                    original_generation_two_ollama = sys.modules.pop('generation_two.ollama', None)
                    
                    try:
                        # More aggressive search: directly search site-packages directories
                        import site
                        import os
                        
                        # Get all site-packages directories
                        site_packages_dirs = site.getsitepackages()
                        if hasattr(site, 'getusersitepackages'):
                            user_site = site.getusersitepackages()
                            if user_site:
                                site_packages_dirs.append(user_site)
                        
                        # Try to find ollama in site-packages
                        ollama_pkg = None
                        list_func = None
                        for site_dir in site_packages_dirs:
                            ollama_path = os.path.join(site_dir, 'ollama')
                            if os.path.exists(ollama_path) or os.path.exists(ollama_path + '.py'):
                                try:
                                    spec = importlib.util.spec_from_file_location(
                                        'ollama_installed',
                                        os.path.join(ollama_path, '__init__.py') if os.path.isdir(ollama_path) else ollama_path + '.py'
                                    )
                                    if spec and spec.loader:
                                        ollama_pkg = importlib.util.module_from_spec(spec)
                                        spec.loader.exec_module(ollama_pkg)
                                        list_func = getattr(ollama_pkg, 'list', None)
                                        if list_func:
                                            sys.modules['_ollama_pkg_installed'] = ollama_pkg
                                            # Try a simple list() call to verify Ollama is actually working
                                            try:
                                                list_func()
                                                logger.info(f"✅ Ollama is actually available! Resetting consecutive_failures from {self.consecutive_failures} to 0")
                                                self.consecutive_failures = 0
                                                # Health check passed - continue to actual generation (don't return None)
                                                break  # Break out of site_dir loop
                                            except Exception as e:
                                                # Ollama library available but server not responding
                                                logger.warning(f"Ollama library available but server not responding: {e}, using fallback")
                                                self.stats['fallback_used'] += 1
                                                return None
                                except Exception as e:
                                    logger.debug(f"Error loading ollama from {site_dir}: {e}")
                                    continue
                        
                        # Fallback: try find_spec
                        if not list_func:
                            spec = importlib.util.find_spec('ollama')
                            if spec and spec.loader and spec.origin:
                                origin_lower = spec.origin.lower()
                                if 'site-packages' in origin_lower or 'dist-packages' in origin_lower:
                                    ollama_pkg = importlib.util.module_from_spec(spec)
                                    spec.loader.exec_module(ollama_pkg)
                                    list_func = getattr(ollama_pkg, 'list', None)
                                    if list_func:
                                        sys.modules['_ollama_pkg_installed'] = ollama_pkg
                                        try:
                                            list_func()
                                            logger.info(f"✅ Ollama is actually available! Resetting consecutive_failures from {self.consecutive_failures} to 0")
                                            self.consecutive_failures = 0
                                            # Health check passed - continue to actual generation (don't return None)
                                        except Exception as e:
                                            logger.warning(f"Ollama library available but server not responding: {e}, using fallback")
                                            self.stats['fallback_used'] += 1
                                            return None
                                else:
                                    # Not in site-packages, but don't raise - try requests fallback instead
                                    logger.debug(f"ollama spec not in site-packages (origin: {spec.origin}), will try requests fallback")
                            else:
                                # No spec found, but don't raise - try requests fallback instead
                                logger.debug(f"ollama spec not found, will try requests fallback")
                        
                        # If list_func was found and called successfully, consecutive_failures should be reset
                        # If not, we'll try requests fallback in the actual generation code
                    finally:
                        # Restore original modules if they existed
                        if original_ollama:
                            sys.modules['ollama'] = original_ollama
                        if original_generation_two_ollama:
                            sys.modules['generation_two.ollama'] = original_generation_two_ollama
                    
                    # Check if health check succeeded (consecutive_failures was reset)
                    if self.consecutive_failures < self.max_consecutive_failures:
                        # Health check passed - continue to actual generation
                        logger.info(f"✅ Health check passed, continuing with Ollama generation (consecutive_failures: {self.consecutive_failures})")
                    else:
                        # Health check failed - but still try generation with requests fallback
                        logger.warning(f"Ollama health check failed, but will still try generation with requests fallback")
                        # Don't return None - let the generation code try requests fallback
                except ImportError as import_err:
                    # Ollama library not available - but still try generation with requests fallback
                    logger.debug(f"Ollama library import failed: {import_err}, but will still try generation with requests fallback")
                    # Don't return None - let the generation code try requests fallback
            
            self.stats['total_requests'] += 1
            logger.debug(f"Lock passed, total_requests: {self.stats['total_requests']}")
            
        # Release lock before making HTTP request to allow concurrent requests
        # Try to generate
        logger.debug(f"Starting Ollama generate() - attempts: {self.max_retries}, model: {self.model}, base_url: {self.base_url}")
        for attempt in range(self.max_retries):
            try:
                # Build messages list (needed for both ollama library and requests fallback)
                messages = []
                if system_prompt:
                    messages.append({
                        'role': 'system',
                        'content': system_prompt
                    })
                messages.append({
                    'role': 'user',
                    'content': prompt
                })
                    
                if progress_callback:
                    try:
                        progress_callback(f"Attempt {attempt + 1}/{self.max_retries}...")
                    except Exception:
                        pass  # Ignore callback errors
                    
                # Start progress monitoring thread
                request_start_time = time.time()
                create_progress_monitor(progress_callback, request_start_time, self.timeout)
                
                # Try ollama library first
                logger.debug(f"Attempt {attempt + 1}: Calling ollama.chat() for model {self.model}")
                generated_text = None
                
                if progress_callback:
                    try:
                        progress_callback("Calling Ollama...")
                    except Exception:
                        pass
                
                chat_func = get_ollama_chat_function()
                if chat_func:
                    generated_text = call_ollama_library(
                        chat_func, self.model, messages, temperature, max_tokens, self.timeout
                    )
                    if generated_text:
                        logger.info(f"Attempt {attempt + 1}: ✅ Ollama returned {len(generated_text)} chars: {generated_text[:100]}...")
                    else:
                        logger.warning(f"Attempt {attempt + 1}: ⚠️ Ollama returned empty response")
                
                # Fallback to requests if ollama didn't work or wasn't available
                if not generated_text:
                    logger.debug(f"Attempt {attempt + 1}: Trying requests fallback to {self.base_url}/api/chat")
                    if progress_callback:
                        try:
                            progress_callback("Using requests fallback...")
                        except Exception:
                            pass
                    
                    # Ensure session exists for requests fallback
                    if not self.session:
                        try:
                            import requests
                            self.session = requests.Session()
                            logger.debug("Created requests session for fallback")
                        except Exception as session_err:
                            logger.error(f"Failed to create requests session: {session_err}")
                            continue  # Skip this attempt
                    
                    generated_text = call_ollama_requests(
                        self.session, self.base_url, self.model, messages, temperature, max_tokens, self.timeout
                    )
                
                if generated_text:
                    # Update stats (quick lock operation)
                    with self.lock:
                        self.stats['successful_requests'] += 1
                        self.consecutive_failures = 0
                    elapsed = time.time() - request_start_time
                    if progress_callback:
                        try:
                            progress_callback(f"✅ Generated ({int(elapsed)}s)")
                        except Exception:
                            pass  # Ignore callback errors
                    # Only log important events, not debug details
                    if elapsed > 30:  # Only log if it took longer than 30s
                        logger.info(f"Ollama generated {len(generated_text)} chars in {int(elapsed)}s")
                    return generated_text
                
                # If we get here, generation failed (empty response)
                # Continue to next attempt
                if attempt < self.max_retries - 1:
                    continue
                    
            except Exception as inner_error:
                # Handle any errors in the inner try block (ollama/requests calls)
                logger.debug(f"Attempt {attempt + 1}: Inner error: {type(inner_error).__name__}: {str(inner_error)[:200]}")
                generated_text = None
                # Continue to next attempt or fall through to outer except
                if attempt < self.max_retries - 1:
                    continue
                    
            except Exception as ollama_error:
                # Handle ollama library errors
                error_msg = str(ollama_error)
                logger.debug(f"Ollama error: {type(ollama_error).__name__}: {error_msg[:200]}")
                
                # Check if it's a model not found error
                if 'model' in error_msg.lower() or 'not found' in error_msg.lower():
                    logger.warning(f"Model '{self.model}' not found in Ollama. Trying to find alternative...")
                    preferred = []
                    if "qwen" in self.model.lower() or "coder" in self.model.lower():
                        preferred = ["qwen2.5-coder:1.5b", "qwen2.5-coder:7b", "qwen2.5-coder:32b"]
                    alternative = self._find_available_model(preferred)
                    if alternative and alternative != self.model:
                        logger.info(f"Switching to available model: {alternative}")
                        self.model = alternative
                        continue  # Retry with new model
                    else:
                        logger.warning(f"Model '{self.model}' not found and no alternative available")
                        self.is_available = False
                
                # Continue to next attempt
                if attempt < self.max_retries - 1:
                    continue
                else:
                    # Last attempt failed
                    if progress_callback:
                        try:
                            progress_callback(f"❌ Error: {error_msg[:50]}")
                        except Exception:
                            pass
                    logger.warning(f"Ollama generation failed after {self.max_retries} attempts: {error_msg[:200]}")
                    
            except requests.exceptions.Timeout:
                error_msg = f"Timeout after {self.timeout}s (attempt {attempt + 1}/{self.max_retries})"
                if progress_callback:
                    try:
                        progress_callback(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignore callback errors
                logger.warning(error_msg)
            except requests.exceptions.ConnectionError as e:
                error_msg = f"Connection error (attempt {attempt + 1}/{self.max_retries}): {str(e)[:100]}"
                if progress_callback:
                    try:
                        progress_callback(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignore callback errors
                logger.warning(error_msg)
                logger.debug(f"Connection error details: {e}", exc_info=True)
                self.is_available = False
            except Exception as e:
                error_msg = f"Error: {str(e)[:100]}"
                if progress_callback:
                    try:
                        progress_callback(f"❌ {error_msg}")
                    except Exception:
                        pass  # Ignore callback errors
                # Log full exception details for debugging
                logger.error(f"Ollama error: {type(e).__name__}: {str(e)[:200]}", exc_info=True)
            
            # All retries failed
        with self.lock:
            self.stats['failed_requests'] += 1
            self.consecutive_failures += 1
            logger.warning("Ollama generation failed after all retries")
            return None
    
    def generate_template(
        self, 
        hypothesis: str,
        region: str = "USA",
        dataset_categories: List[str] = None,
        avoid_duplicates_context: str = "",
        available_operators: List[Dict] = None,
        available_fields: List[Dict] = None,
        successful_patterns: List[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        use_placeholder_fields: bool = True,  # V2 approach: use placeholders to avoid misspelling
        forbidden_operators: List[str] = None  # Operators that are forbidden (already used in batch)
    ) -> Optional[str]:
        """
        Generate alpha template from hypothesis with enhanced prompt engineering and AST guidance
        
        Args:
            hypothesis: Research hypothesis
            region: Region code
            dataset_categories: Required dataset categories (for themes)
            avoid_duplicates_context: Context string with expressions to avoid
            available_operators: List of available operators from operatorRAW.json
            available_fields: List of available data fields for the region
            successful_patterns: List of successful template patterns to guide generation
            
        Returns:
            Alpha expression or None
        """
        system_prompt = """You are an expert in quantitative finance and WorldQuant Brain alpha generation.
Generate alpha expressions in FASTEXPR format that are syntactically correct and follow WorldQuant Brain conventions.
FASTEXPR combines OPERATORS (functions) and DATA FIELDS (variables) in specific patterns.
Return only the alpha expression, no explanations or markdown."""
        
        user_prompt = f"""Generate a WorldQuant Brain FASTEXPR alpha expression for the following hypothesis:
{hypothesis}

Region: {region}

CRITICAL SIMPLICITY RULE:
- Use ONLY 1-5 operators total. DO NOT create deeply nested expressions.
- Keep it simple: operator(field) or operator1(operator2(field1) field2) is good.
- AVOID excessive nesting like operator1(operator2(operator3(operator4(operator5(...))))) - this is WRONG.
- Simple expressions perform better and are easier to understand.

CRITICAL FASTEXPR SYNTAX RULES:
1. FASTEXPR uses OPERATOR(FIELD_ID, parameters) syntax - operators are functions, field IDs are variable identifiers
2. Field IDs are identifiers like "anl49_1stfiscalquarterearningspershare" - they do NOT have region prefix like "USA.MCAP"
3. Use the EXACT field ID from the available fields list - do NOT add region prefix (NO "USA.", "EUR.", "CHN.", etc.)
4. NEVER use region prefixes: "USA.MCAP" is WRONG, use "anl14_actvalue_capex_fp0" or similar field ID directly
5. NEVER use leading + signs: "+field" is WRONG, use "field" directly
6. NEVER use * operator alone: "*(field" is WRONG, use "multiply(field" or just "field"
7. OPERATORS WITH PARAMETERS USE COMMAS: "operator(field, param)" is CORRECT
   - Example: ts_rank(DATA_FIELD1, 20) - comma between field and parameter
   - Example: winsorize(DATA_FIELD1, 4) - comma between field and parameter
   - Example: rank(DATA_FIELD1, DATA_FIELD2) - comma between multiple fields
   - Arithmetic expressions: DATA_FIELD1 + DATA_FIELD2 - NO comma (just space around +)
8. Operator scope compatibility:
   - REGULAR scope operators work with REGULAR, MATRIX, and VECTOR fields
   - MATRIX scope operators work with MATRIX fields
   - VECTOR scope operators work with VECTOR fields
9. Time series operators (ts_rank, ts_delta, ts_mean) typically work with MATRIX fields
10. Cross-sectional operators (rank, delta) typically work with REGULAR fields
11. Arithmetic operators: +, -, *, /, ^, %, >, <, >=, <=, ==, !=, &&, ||
12. All parentheses must be balanced
13. Field IDs are case-sensitive and must match EXACTLY from the available fields list"""
        
        # V2 Approach: Use ONLY the provided selected operators (exclusive selection from caller)
        if available_operators:
            # Use the provided operators directly (they are pre-selected for diversity by caller)
            selected_operators = available_operators  # Already filtered by caller for exclusive selection
            
            # Show operators with clear instructions to use ONLY these
            operator_list = []
            operator_names = []
            for i, op in enumerate(selected_operators):
                name = op.get('name', '')
                definition = op.get('definition', '')
                category = op.get('category', '')
                if name:
                    operator_names.append(name)
                    if definition:
                        operator_list.append(f"[{i}] {name}: {definition} ({category})")
                    else:
                        operator_list.append(f"[{i}] {name} ({category})")
            
            if operator_list:
                user_prompt += "\n\n🚨 CRITICAL: AVAILABLE OPERATORS - USE ONLY THESE OPERATORS:"
                user_prompt += "\n" + "\n".join(operator_list)
                user_prompt += f"\n\n🚨 EXCLUSIVE OPERATOR LIST: {', '.join(operator_names)}"
                
                # Add forbidden operators list if provided
                if forbidden_operators:
                    user_prompt += f"\n\n🚫 FORBIDDEN OPERATORS (DO NOT USE THESE - already used in batch): {', '.join(forbidden_operators)}"
                    user_prompt += "\n⚠️ If you use any of these forbidden operators, your expression will be REJECTED!"
                    user_prompt += "\n⚠️ Use ONLY operators from the AVAILABLE OPERATORS list above, NOT from the FORBIDDEN list!"
                
                user_prompt += "\n\nCRITICAL RULES:"
                user_prompt += "\n1. You MUST use ONLY operators from the AVAILABLE OPERATORS list above"
                if forbidden_operators:
                    user_prompt += "\n2. DO NOT use operators from the FORBIDDEN list (if shown above)"
                    user_prompt += "\n3. DO NOT use operators that are NOT in the AVAILABLE list (e.g., if 'rank' is not listed, DO NOT use 'rank')"
                else:
                    user_prompt += "\n2. DO NOT use operators that are NOT in the list (e.g., if 'rank' is not listed, DO NOT use 'rank')"
                user_prompt += "\n4. If you see 'rank' in the AVAILABLE list, you can use it. If NOT, use alternatives like 'ts_rank', 'winsorize', 'zscore', etc."
                user_prompt += "\n5. Mix different operator types from the AVAILABLE OPERATORS list"
                user_prompt += "\n6. Use actual operator names exactly as shown in the AVAILABLE OPERATORS list above"
        
        # V2 Approach: Use placeholders (DATA_FIELD1, DATA_FIELD2, etc.) to avoid misspelling
        if use_placeholder_fields and available_fields:
            # Show fields with indices for reference, but instruct to use placeholders
            field_list = []
            for i, field in enumerate(available_fields[:30]):  # Show first 30 fields
                field_id = field.get('id', '')
                if field_id:
                    field_list.append(f"[{i}] {field_id}")
            
            if field_list:
                user_prompt += "\n\nAVAILABLE DATA FIELDS (use placeholders DATA_FIELD1, DATA_FIELD2, etc. - do NOT use actual field names!):"
                user_prompt += "\n" + "\n".join(field_list[:20])  # Show first 20
                user_prompt += f"\n... and {len(available_fields) - 20} more fields available"
                user_prompt += "\n\nCRITICAL: Use PLACEHOLDERS, not actual field names!"
                user_prompt += "\n- Use DATA_FIELD1, DATA_FIELD2, DATA_FIELD3, DATA_FIELD4 instead of actual field IDs"
                user_prompt += "\n- Example: ts_rank(DATA_FIELD1, 20) NOT ts_rank(anl14_actvalue_capex_fp0, 20)"
                user_prompt += "\n- Example: winsorize(DATA_FIELD1 + DATA_FIELD2, 4) NOT winsorize(anl14_actvalue_capex_fp0 + anl14_actvalue_bvps_fp0, 4)"
                user_prompt += "\n- This prevents misspelling errors - placeholders will be replaced automatically"
        elif available_fields:
            # Fallback: Show actual field IDs (old approach)
            field_examples = []
            for field in available_fields:
                field_id = field.get('id', '')
                if field_id:
                    field_examples.append(f"  - {field_id}")
                    if len(field_examples) >= 15:
                        break
            
            if field_examples:
                user_prompt += "\n\nAVAILABLE DATA FIELDS (use these EXACT field IDs - they don't have region prefix):"
                user_prompt += "\n" + "\n".join(field_examples)
                user_prompt += f"\n\nNOTE: Field IDs are identifiers like 'anl49_1stfiscalquarterearningspershare', NOT 'USA.MCAP'."
                user_prompt += f"\nUse the field ID directly in expressions, e.g., ts_rank({field_examples[0].strip().replace('  - ', '')}, 20)"
        
        # Add successful patterns as examples
        if successful_patterns:
            user_prompt += "\n\nSUCCESSFUL PATTERN EXAMPLES (learn from these structures):"
            for pattern in successful_patterns[:3]:  # Show top 3 patterns
                user_prompt += f"\n  - {pattern}"
        
        # Add concrete syntax examples using placeholders (V2 approach) - WITH COMMAS for parameters
        user_prompt += "\n\nDIVERSE SYNTAX EXAMPLES (use placeholders DATA_FIELD1, DATA_FIELD2, etc. - COMMAS for parameters!):"
        if use_placeholder_fields:
            # Use placeholders in examples - COMMAS for operator parameters
            user_prompt += "\n  - winsorize(DATA_FIELD1, 4)  # Winsorize to 4 std - COMMA between field and parameter"
            user_prompt += "\n  - zscore(DATA_FIELD1)  # Z-score normalization (no parameter, no comma needed)"
            user_prompt += "\n  - rank(DATA_FIELD1)  # Cross-sectional rank (no parameter, no comma needed)"
            user_prompt += "\n  - ts_rank(DATA_FIELD1, 20)  # Time series rank - COMMA between field and parameter"
            user_prompt += "\n  - DATA_FIELD1 + DATA_FIELD2  # Arithmetic addition - NO comma (just space around +)"
            user_prompt += "\n  - power(DATA_FIELD1, 2)  # Power operator - COMMA between field and parameter"
            user_prompt += "\n  - abs(DATA_FIELD1 - ts_mean(DATA_FIELD1, 20))  # Absolute deviation - COMMA in ts_mean"
            user_prompt += "\n  - ts_rank(DATA_FIELD1 * DATA_FIELD2, 10)  # Rank of product - COMMA before parameter"
            user_prompt += "\n  - winsorize(DATA_FIELD1 / DATA_FIELD2, 3)  # Winsorized ratio - COMMA before parameter"
            user_prompt += "\n  - rank(normalize(log(DATA_FIELD1)), DATA_FIELD2)  # Complex nested - COMMA between arguments"
        elif available_fields and len(available_fields) > 0:
            # Fallback: Use actual field IDs (old approach)
            example_fields = [f.get('id', '') for f in available_fields[:6] if f.get('id')]
            if example_fields:
                user_prompt += f"\n  - winsorize({example_fields[0]}, 4)  # Winsorize to 4 std"
                if len(example_fields) > 1:
                    user_prompt += f"\n  - zscore({example_fields[1]})  # Z-score normalization"
                if len(example_fields) > 2:
                    user_prompt += f"\n  - rank({example_fields[2]})  # Cross-sectional rank"
                if len(example_fields) > 3:
                    user_prompt += f"\n  - {example_fields[3]} + {example_fields[4]}  # Arithmetic addition"
        else:
            # Fallback examples
            user_prompt += "\n  - winsorize(field_id, 4)  # Winsorize to 4 std"
            user_prompt += "\n  - zscore(field_id)  # Z-score normalization"
            user_prompt += "\n  - rank(field_id)  # Cross-sectional rank"
        
        if dataset_categories:
            user_prompt += f"\n\nRequired dataset categories: {', '.join(dataset_categories)}"
            user_prompt += "\nExcluded datasets: imbalance5, model110, pv1, other335, model39"
            user_prompt += "\nNote: You can use grouping fields from pv1: country, exchange, market, sector, industry, subindustry"
        
        # Add duplicate avoidance context if provided
        if avoid_duplicates_context:
            user_prompt += f"\n\n{avoid_duplicates_context}"
            user_prompt += "\nGenerate a NEW and DIFFERENT expression, not similar to the ones above."
        
        user_prompt += "\n\nDIVERSITY REQUIREMENTS:"
        user_prompt += "\n- Use DIFFERENT operators each time (avoid repeating ts_rank, ts_rank, ts_rank...)"
        user_prompt += "\n- Mix arithmetic operators: +, -, *, /, ^, power, signed_power, abs, log, sqrt"
        user_prompt += "\n- Use cross-sectional operators: winsorize, zscore, rank, delta"
        user_prompt += "\n- Combine operators creatively: e.g., winsorize(DATA_FIELD1 + DATA_FIELD2, 4) or power(abs(DATA_FIELD1), 0.5)"
        user_prompt += "\n- Use logical operators when appropriate: >, <, >=, <=, ==, !=, &&, ||"
        user_prompt += "\n- CRITICAL: Use COMMAS for operator parameters: ts_rank(DATA_FIELD1, 20) NOT ts_rank(DATA_FIELD1 20)"
        
        user_prompt += "\n\n🚨 CRITICAL: MAXIMUM 5 OPERATORS - NO EXCEPTIONS!"
        user_prompt += "\n- Count operators carefully: ts_rank, rank, normalize, log, abs, winsorize, etc. - each counts as 1 operator"
        user_prompt += "\n- Arithmetic operators (+, -, *, /) also count as operators"
        user_prompt += "\n- Example with 3 operators: rank(normalize(log(DATA_FIELD1))) - rank(1) + normalize(1) + log(1) = 3 operators"
        user_prompt += "\n- Example with 5 operators: ts_rank(winsorize(normalize(log(abs(DATA_FIELD1))), 4), 20) - exactly 5 operators"
        user_prompt += "\n- DO NOT exceed 5 operators - if you need more, simplify the expression!"
        user_prompt += "\n\n🚨 CRITICAL: NO CONSECUTIVE DUPLICATE OPERATORS!"
        user_prompt += "\n- DO NOT nest the same operator multiple times: ts_step(ts_step(ts_step(...))) is FORBIDDEN"
        user_prompt += "\n- DO NOT use: ts_count_nans(ts_count_nans(ts_count_nans(...))) - this is FORBIDDEN"
        user_prompt += "\n- DO NOT use: hump(hump(hump(...))) - this is FORBIDDEN"
        user_prompt += "\n- Each operator should appear only ONCE in the expression, or at most in different contexts (not nested)"
        user_prompt += "\n- Example of FORBIDDEN: ts_step(ts_step(ts_step(field))) - REJECTED"
        user_prompt += "\n- Example of ALLOWED: ts_step(field, 20) + ts_step(field, 10) - different contexts, OK"
        user_prompt += "\n\nGenerate a SIMPLE, valid FASTEXPR expression using MAXIMUM 5 operators total. Keep it simple - avoid deep nesting."
        user_prompt += "\nUse COMMAS for parameters: operator(field, param). Return ONLY the expression:"
        user_prompt += "\n\n🚨 CRITICAL: Return ONLY the FASTEXPR expression. NO natural language, NO explanations, NO 'Let's generate...', NO 'We'll focus...'."
        user_prompt += "\nJust return the pure expression like: rank(normalize(log(DATA_FIELD1)), 4)"
        
        # Debug: Log that we're about to call generate
        logger.debug(f"Calling Ollama generate() with prompt length: {len(user_prompt)}")
        result = self.generate(user_prompt, system_prompt, temperature=0.7, max_tokens=300, progress_callback=progress_callback)
        logger.debug(f"Ollama generate() returned: {result[:50] if result else 'None'}...")
        
        if result:
            # Clean up the result
            result = result.strip()
            # Remove markdown code blocks if present
            if '```' in result:
                lines = result.split('\n')
                result = '\n'.join([l for l in lines if not l.strip().startswith('```')])
            
            # Remove common explanatory prefixes
            explanatory_prefixes = [
                'Here is the corrected FASTEXPR alpha expression:',
                'Here is the expression:',
                'The expression is:',
                'Corrected expression:',
                'The corrected expression is:',
                'FASTEXPR expression:',
                'Alpha expression:',
                'Expression:',
                'Alpha:',
                'FASTEXPR:',
                'The expression',
                'Here is the expression',
            ]
            
            for prefix in explanatory_prefixes:
                if result.lower().startswith(prefix.lower()):
                    result = result[len(prefix):].strip()
                    result = result.lstrip(':').strip()
            
            # Remove backticks (common error - they shouldn't be in FASTEXPR)
            result = result.replace('`', '')
            
            # Look for expression-like patterns (contains parentheses and operators)
            lines = result.split('\n')
            expression_candidates = []
            for line in lines:
                line = line.strip()
                # Remove backticks
                line = line.replace('`', '')
                # Check if line looks like an expression (has parentheses and alphanumeric)
                if '(' in line and ')' in line and any(c.isalnum() or c in '._' for c in line):
                    # Remove trailing punctuation
                    line = line.rstrip('.,;!?')
                    # Remove any remaining backticks
                    line = line.replace('`', '')
                    expression_candidates.append(line)
            
            # Remove natural language prefixes more aggressively
            # Look for patterns like "Let's generate...", "We'll focus...", etc.
            import re
            # Remove lines that start with natural language patterns
            natural_language_patterns = [
                r'^let\'?s\s+',
                r'^we\'?ll\s+',
                r'^we\s+will\s+',
                r'^generate\s+a\s+',
                r'^create\s+a\s+',
                r'^build\s+a\s+',
                r'^focus\s+on\s+',
                r'^using\s+maximum\s+',
                r'^we\'?re\s+',
            ]
            cleaned_lines = []
            for line in lines:
                line_stripped = line.strip()
                # Skip lines that are pure natural language
                is_natural_language = False
                for pattern in natural_language_patterns:
                    if re.match(pattern, line_stripped, re.IGNORECASE):
                        is_natural_language = True
                        break
                if not is_natural_language:
                    cleaned_lines.append(line)
            
            if cleaned_lines:
                result = '\n'.join(cleaned_lines)
            
            # If we found expression candidates, use the first one
            if expression_candidates:
                result = expression_candidates[0]
            else:
                # Fallback: take first line and clean it
                result = lines[0].strip()
                result = result.rstrip('.,;!?')
                # Remove backticks
                result = result.replace('`', '')
            
            # Remove quotes
            result = result.strip('"\'')
            # Final cleanup: remove any remaining backticks
            result = result.replace('`', '')
            
            # Additional cleanup: remove region prefixes and fix syntax
            import re
            # Remove region prefixes (USA., EUR., CHN., etc.)
            result = re.sub(r'\b(USA|EUR|CHN|ASI|GLB|IND)\.([a-z][a-z0-9_]+)\b', r'\2', result, flags=re.IGNORECASE)
            # Remove leading + signs
            result = re.sub(r'\(\s*\+([a-z][a-z0-9_]+)', r'(\1', result, flags=re.IGNORECASE)
            result = re.sub(r'\(\s*\+([a-z_]+)\s*\(', r'(\1(', result, flags=re.IGNORECASE)
            # Fix invalid * operator
            result = re.sub(r'\(\s*\*\s*\(([a-z][a-z0-9_]+)', r'(\1', result, flags=re.IGNORECASE)
            result = re.sub(r'\(\s*\*\s*([a-z][a-z0-9_]+)', r'(\1', result, flags=re.IGNORECASE)
            # Remove invalid commas
            result = re.sub(r'([a-z0-9_\)])\s*,\s*([a-z0-9_\(])', r'\1 \2', result, flags=re.IGNORECASE)
            result = re.sub(r'([a-z0-9_\)])\s*,\s*(\d)', r'\1 \2', result, flags=re.IGNORECASE)
            result = re.sub(r'(\d)\s*,\s*([a-z0-9_\(])', r'\1 \2', result, flags=re.IGNORECASE)
            # Fix double operators
            result = re.sub(r'([a-z_]+)\1', r'\1', result, flags=re.IGNORECASE)
            # Clean up spaces
            result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def get_stats(self) -> Dict:
        """Get manager statistics"""
        return {
            **self.stats,
            'is_available': self.is_available,
            'consecutive_failures': self.consecutive_failures,
            'success_rate': (
                self.stats['successful_requests'] / self.stats['total_requests']
                if self.stats['total_requests'] > 0 else 0.0
            )
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'fallback_used': 0
        }

