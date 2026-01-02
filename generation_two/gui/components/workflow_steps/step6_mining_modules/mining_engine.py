"""
Mining Engine Module
Core mining logic with self-sustaining operation
"""

import logging
import time
import threading
from typing import Optional, Callable
from generation_two.core.mining import MiningCoordinator, SearchStrategy
from generation_two.core.slot_manager import SlotManager, SlotStatus
from generation_two.core.simulator_tester import SimulatorTester, SimulationSettings
from generation_two.core.region_config import REGION_DEFAULT_UNIVERSE, REGION_DEFAULT_NEUTRALIZATION

logger = logging.getLogger(__name__)


class MiningEngine:
    """
    Core mining engine with self-sustaining operation
    Handles generation, simulation, and error recovery
    """
    
    def __init__(
        self,
        generator,
        simulator_tester,
        backtest_storage,
        slot_manager: SlotManager,
        correlation_tracker,
        duplicate_detector,
        search_strategy_manager,
        sim_counter,
        log_callback: Optional[Callable[[str], None]] = None,
        update_slot_callback: Optional[Callable] = None
    ):
        """
        Initialize mining engine
        
        Args:
            generator: Template generator
            simulator_tester: Simulator tester
            backtest_storage: Backtest storage
            slot_manager: Slot manager
            correlation_tracker: Correlation tracker
            duplicate_detector: Duplicate detector
            search_strategy_manager: Search strategy manager
            sim_counter: Simulation counter
            log_callback: Log callback
            update_slot_callback: Slot update callback
        """
        self.generator = generator
        self.simulator_tester = simulator_tester
        self.backtest_storage = backtest_storage
        self.slot_manager = slot_manager
        self.correlation_tracker = correlation_tracker
        self.duplicate_detector = duplicate_detector
        self.search_strategy = search_strategy_manager
        self.sim_counter = sim_counter
        self.log_callback = log_callback
        self.update_slot_callback = update_slot_callback
        
        self.mining_active = False
        self.stop_flag = False
        self.generated_templates_queue = []
        self.error_recovery_count = 0
        self.max_error_recovery = 10  # Max consecutive errors before pause
    
    def start(self):
        """Start mining engine"""
        if self.mining_active:
            return
        
        self.mining_active = True
        self.stop_flag = False
        self.error_recovery_count = 0
        
        # Start main loop
        thread = threading.Thread(target=self._main_loop, daemon=True, name="MiningEngine")
        thread.start()
        
        self._log("✅ Mining engine started")
    
    def stop(self):
        """Stop mining engine"""
        self.mining_active = False
        self.stop_flag = True
        self._log("⏹ Mining engine stopping...")
    
    def _main_loop(self):
        """Main mining loop with error recovery"""
        while self.mining_active and not self.stop_flag:
            try:
                # Check simulation limit
                if not self.sim_counter.can_simulate():
                    self._log("⚠️ Daily simulation limit reached. Waiting...")
                    time.sleep(3600)
                    continue
                
                # Generate templates if queue is low
                if len(self.generated_templates_queue) < 10:
                    self._generate_templates_batch()
                
                # Process simulations
                self._process_simulations()
                
                # Reset error recovery on success
                self.error_recovery_count = 0
                
                time.sleep(1)
                
            except Exception as e:
                self.error_recovery_count += 1
                logger.error(f"Mining engine error (recovery {self.error_recovery_count}): {e}", exc_info=True)
                self._log(f"❌ Error: {str(e)[:100]} (recovery {self.error_recovery_count}/{self.max_error_recovery})")
                
                if self.error_recovery_count >= self.max_error_recovery:
                    self._log("⚠️ Too many errors, pausing for 60 seconds...")
                    time.sleep(60)
                    self.error_recovery_count = 0
                else:
                    time.sleep(5)  # Brief pause before retry
    
    def _generate_templates_batch(self, batch_size: int = 5):
        """Generate a batch of templates"""
        region = self.search_strategy.get_next_region()
        if not region:
            return
        
        templates_generated = 0
        for _ in range(batch_size):
            if self.stop_flag:
                break
            
            try:
                # Generate template
                template = self.generator.template_generator.ollama_manager.generate_template(
                    hypothesis=f"Generate a WorldQuant Brain FASTEXPR alpha expression for {region} region.",
                    region=region,
                    available_operators=self.generator.template_generator.operator_fetcher.operators if self.generator.template_generator.operator_fetcher else None,
                    available_fields=self.generator.template_generator.get_data_fields_for_region(region)
                )
                
                if template:
                    template = template.replace('`', '').strip()
                    
                    # Replace field placeholders (DATA_FIELD1, DATA_FIELD2, etc.) with actual field IDs
                    available_fields = self.generator.template_generator.get_data_fields_for_region(region)
                    if available_fields and ('DATA_FIELD' in template.upper() or 'data_field' in template.lower()):
                        template = self.generator.template_generator._replace_field_placeholders(
                            template, available_fields, region
                        )
                        logger.debug(f"Replaced placeholders in template: {template[:100]}...")
                    
                    # Check duplicates
                    is_dup, reason = self.duplicate_detector.is_duplicate(template, region)
                    if is_dup:
                        self._log(f"⚠️ Duplicate filtered: {reason}")
                        continue
                    
                    # Add to queue
                    self.generated_templates_queue.append((template, region))
                    templates_generated += 1
                    
            except Exception as e:
                logger.debug(f"Error generating template: {e}")
                continue
        
        if templates_generated > 0:
            self._log(f"✅ Generated {templates_generated} templates for {region}")
    
    def _process_simulations(self):
        """Process pending simulations"""
        if not self.generated_templates_queue:
            return
        
        # Select low-correlation templates
        candidates = self.generated_templates_queue[:20]
        low_corr_templates = self.correlation_tracker.get_low_correlation_templates(
            candidates,
            max_correlation=0.3,
            limit=8
        )
        
        # Prepare simulation batch
        if low_corr_templates:
            selected = [(t, r) for t, r, _ in low_corr_templates]
        else:
            selected = self.generated_templates_queue[:8]
        
        # Remove from queue
        for template, region in selected:
            if (template, region) in self.generated_templates_queue:
                self.generated_templates_queue.remove((template, region))
        
        # Submit simulations
        for template, region in selected:
            if self.stop_flag:
                break
            
            # Check simulation limit
            status = self.sim_counter.increment_count()
            if not status['can_simulate']:
                self._log("⚠️ Daily simulation limit reached")
                break
            
            # Assign slot (GLB uses 2 slots)
            slot_count = 2 if region == 'GLB' else 1
            slot_ids = self.slot_manager.find_available_slots(slot_count)
            
            if not slot_ids:
                # Wait for slots
                wait_count = 0
                while wait_count < 10 and not self.stop_flag:
                    time.sleep(2)
                    wait_count += 1
                    slot_ids = self.slot_manager.find_available_slots(slot_count)
                    if slot_ids:
                        break
                
                if not slot_ids:
                    self._log(f"⚠️ No slots available for {region}")
                    continue
            
            # Assign slots
            assigned_slots = self.slot_manager.assign_slot(template, region, 0)
            if not assigned_slots:
                continue
            
            # Start simulation in thread
            for slot_id in assigned_slots:
                thread = threading.Thread(
                    target=self._run_simulation,
                    args=(slot_id, template, region),
                    daemon=True
                )
                thread.start()
            
            time.sleep(0.5)  # Brief pause between submissions
    
    def _run_simulation(self, slot_id: int, template: str, region: str):
        """Run a single simulation"""
        try:
            settings = SimulationSettings(
                universe=REGION_DEFAULT_UNIVERSE.get(region, 'TOP3000'),
                neutralization=REGION_DEFAULT_NEUTRALIZATION.get(region, 'INDUSTRY'),
                delay=1,
                testPeriod="P5Y0M0D"
            )
            
            # Update slot
            slot = self.slot_manager.get_slot_status(slot_id)
            if slot:
                slot.add_log(f"[{region}] Submitting...")
            self.slot_manager.update_slot_progress(slot_id, percent=10, message="Submitting...", api_status="PENDING")
            self._update_slot(slot_id, template, region, 10, "Submitting...")
            
            # Submit
            progress_url = self.simulator_tester.submit_simulation(template, region, settings)
            if not progress_url:
                self.slot_manager.release_slot(slot_id, success=False, error="Failed to submit")
                return
            
            # Monitor
            def progress_callback(percent, message, api_status):
                self.slot_manager.update_slot_progress(slot_id, percent=percent, message=message, api_status=api_status)
                slot = self.slot_manager.get_slot_status(slot_id)
                if slot:
                    slot.add_log(f"[{api_status}] {message}")
                self._update_slot(slot_id, template, region, percent, message)
            
            result = self.simulator_tester.monitor_simulation(
                progress_url, template, region, settings,
                progress_callback=progress_callback
            )
            
            # Handle refeed if failed
            if not result.success:
                result = self._handle_refeed(slot_id, template, region, result.error_message, settings)
            
            # Save result
            if self.backtest_storage:
                self.backtest_storage.store_result(result)
            
            # Update correlation tracker
            if result.success and result.alpha_id:
                self.correlation_tracker.update_template_alpha_mapping(template, str(result.alpha_id))
                self.search_strategy.add_successful_template(template, region)
            
            # Release slot
            self.slot_manager.release_slot(
                slot_id,
                success=result.success,
                result={
                    'sharpe': result.sharpe,
                    'fitness': result.fitness,
                    'alpha_id': str(result.alpha_id) if result.alpha_id else ""
                } if result.success else None,
                error=result.error_message if not result.success else None
            )
            
            # Update display
            status = "SUCCESS" if result.success else "FAILED"
            message = f"Sharpe: {result.sharpe:.2f}" if result.success else (result.error_message[:30] if result.error_message else "Failed")
            self._update_slot(slot_id, template, region, 100 if result.success else 0, message, status)
            
            # Log
            if result.success:
                self._log(f"✅ {region}: Sharpe={result.sharpe:.2f}, Fitness={result.fitness:.2f}")
            else:
                self._log(f"❌ {region}: {result.error_message[:50] if result.error_message else 'Unknown error'}")
                
        except Exception as e:
            logger.error(f"Simulation error: {e}", exc_info=True)
            self.slot_manager.release_slot(slot_id, success=False, error=str(e))
            self._update_slot(slot_id, template, region, 0, str(e)[:30], "FAILED")
    
    def _handle_refeed(self, slot_id: int, template: str, region: str, error_message: str, settings: SimulationSettings):
        """Handle refeed correction"""
        if not self.generator.template_generator.template_validator:
            return None
        
        slot = self.slot_manager.get_slot_status(slot_id)
        if slot:
            slot.add_log("🔄 Attempting refeed correction...")
        self._update_slot(slot_id, template, region, 50, "Fixing template...")
        
        # Check if event input error (unlimited retries)
        is_event_input_error = 'does not support event inputs' in error_message.lower()
        max_attempts = 999 if is_event_input_error else 3
        
        fixed_template, fixes = self.generator.template_generator.template_validator.refeed_with_correction(
            template, error_message, region, max_attempts=max_attempts
        )
        
        if fixed_template and fixed_template != template:
            # Retry with fixed template
            slot = self.slot_manager.get_slot_status(slot_id)
            if slot:
                slot.add_log(f"✅ Fixed with {len(fixes)} corrections, retrying...")
            progress_url = self.simulator_tester.submit_simulation(fixed_template, region, settings)
            
            if progress_url:
                def progress_callback(percent, message, api_status):
                    self.slot_manager.update_slot_progress(slot_id, percent=percent, message=message, api_status=api_status)
                    slot = self.slot_manager.get_slot_status(slot_id)
                    if slot:
                        slot.add_log(f"[{api_status}] {message}")
                    self._update_slot(slot_id, fixed_template, region, percent, message)
                
                result = self.simulator_tester.monitor_simulation(
                    progress_url, fixed_template, region, settings,
                    progress_callback=progress_callback
                )
                return result
        
        return None
    
    def _update_slot(self, slot_id: int, template: str, region: str, progress: float, message: str, status: str = "RUNNING"):
        """Update slot display"""
        if self.update_slot_callback:
            slot = self.slot_manager.get_slot_status(slot_id)
            logs = slot.get_logs()[-5:] if slot else []
            self.update_slot_callback(
                slot_id, status, template[:40] + "..." if len(template) > 40 else template,
                f"Region: {region}", progress, message, logs
            )
    
    def _log(self, message: str):
        """Log message"""
        logger.info(message)
        if self.log_callback:
            self.log_callback(message)
