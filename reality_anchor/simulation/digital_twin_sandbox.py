"""
Digital Twin Sandbox for Aegis Nexus
Physics-based simulation environment for consequence prediction.

This module implements a digital twin system that simulates the consequences
of proposed actions before real-world execution.
"""

import asyncio
import logging
import copy
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SimulationState:
    """Represents the state of the digital twin at a point in time"""
    timestamp: datetime
    variables: Dict[str, Any]
    metrics: Dict[str, float]
    constraints: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SimulationResult:
    """Results from a digital twin simulation"""
    simulation_id: str
    initial_state: SimulationState
    final_state: SimulationState
    timeline: List[SimulationState]
    risk_assessment: Dict[str, Any]
    safety_violations: List[str]
    performance_metrics: Dict[str, float]
    financial_impact: Dict[str, float]  # Added Blast Radius Analysis
    recommendations: List[str]
    confidence: float  # 0.0 to 1.0

@dataclass
class InterventionPlan:
    """Plan for an intervention to be simulated"""
    action_id: str
    action_description: str
    target_variables: List[str]
    expected_effects: Dict[str, Any]
    duration_seconds: float
    resource_requirements: List[str]
    safety_constraints: List[str]

class DigitalTwinSimulator:
    """
    Digital twin simulation environment for consequence prediction.
    
    Creates physics-based simulations of system states and predicts
    the outcomes of proposed interventions.
    """
    
    def __init__(self, simulation_granularity: float = 1.0):
        """
        Initialize the digital twin simulator.
        
        Args:
            simulation_granularity: Time step in seconds for simulation
        """
        self.simulation_granularity = simulation_granularity
        self.system_models = {}
        self.physical_laws = {}
        self.safety_constraints = {}
        self.simulation_history: List[SimulationResult] = []
        self.max_history_size = 100
        
    async def initialize(self) -> bool:
        """Initialize the digital twin simulator."""
        try:
            logger.info("🎮 Initializing digital twin simulator...")
            
            # Initialize system models
            await self._initialize_system_models()
            
            # Load physical laws and constraints
            await self._load_physical_laws()
            
            # Initialize safety constraints
            await self._initialize_safety_constraints()
            
            logger.info("✅ Digital twin simulator initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize digital twin simulator: {str(e)}")
            return False
    
    async def _initialize_system_models(self):
        """Initialize models for different system components."""
        # Server/Infrastructure model
        self.system_models['server_infrastructure'] = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'network_bandwidth': 0.0,
            'temperature': 25.0,
            'power_consumption': 0.0
        }
        
        # Database model
        self.system_models['database'] = {
            'connections': 0,
            'query_rate': 0.0,
            'storage_usage': 0.0,
            'backup_status': 'ok',
            'replication_lag': 0.0
        }
        
        # Network model
        self.system_models['network'] = {
            'latency': 0.0,
            'packet_loss': 0.0,
            'bandwidth_utilization': 0.0,
            'active_connections': 0,
            'security_events': 0
        }
        
        # Application model
        self.system_models['application'] = {
            'active_users': 0,
            'requests_per_second': 0.0,
            'error_rate': 0.0,
            'response_time': 0.0,
            'availability': 1.0
        }
    
    async def _load_physical_laws(self):
        """Load physical laws governing system behavior."""
        # CPU utilization dynamics
        self.physical_laws['cpu_dynamics'] = {
            'utilization_rate': 0.1,  # Rate of change per second
            'cooling_rate': 0.05,     # Cooling rate when idle
            'thermal_coupling': 0.02  # Heat transfer coefficient
        }
        
        # Memory allocation dynamics
        self.physical_laws['memory_dynamics'] = {
            'allocation_rate': 0.05,
            'garbage_collection_efficiency': 0.8,
            'fragmentation_growth': 0.01
        }
        
        # Network flow dynamics
        self.physical_laws['network_dynamics'] = {
            'bandwidth_saturation': 0.95,
            'congestion_growth': 0.1,
            'latency_increase_rate': 0.05
        }
        
        # Database performance laws
        self.physical_laws['database_dynamics'] = {
            'query_performance_scaling': -0.5,  # Negative correlation with load
            'storage_growth_rate': 0.02,
            'backup_overhead': 0.1
        }
    
    async def _initialize_safety_constraints(self):
        """Initialize safety constraints for simulation validation."""
        self.safety_constraints = {
            'cpu_max': 0.95,           # 95% CPU utilization maximum
            'memory_max': 0.90,        # 90% memory utilization maximum
            'disk_min_free': 0.10,     # 10% minimum free disk space
            'temperature_max': 75.0,   # 75°C maximum temperature
            'network_latency_max': 1000.0,  # 1000ms maximum latency
            'error_rate_max': 0.05,    # 5% maximum error rate
            'availability_min': 0.99   # 99% minimum availability
        }
    
    async def simulate_intervention(self, intervention_plan: InterventionPlan) -> SimulationResult:
        """
        Simulate the consequences of an intervention.
        
        Args:
            intervention_plan: The intervention to simulate
            
        Returns:
            SimulationResult with predicted outcomes
        """
        logger.info(f"🎮 Simulating intervention: {intervention_plan.action_description}")
        
        try:
            # Create simulation ID
            simulation_id = f"sim_{intervention_plan.action_id}_{int(datetime.utcnow().timestamp())}"
            
            # Get current system state
            current_state = await self._get_current_system_state()
            
            # Clone state for simulation
            simulation_state = copy.deepcopy(current_state)
            
            # Apply intervention effects
            modified_state = await self._apply_intervention_effects(
                simulation_state, intervention_plan
            )
            
            # Run simulation over time
            timeline = await self._run_temporal_simulation(
                modified_state, intervention_plan.duration_seconds
            )
            
            # Assess risks and safety violations
            risk_assessment = await self._assess_risks(timeline)
            safety_violations = await self._check_safety_violations(timeline)
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(timeline)
            
            # Calculate Financial BLAST RADIUS
            financial_impact = await self._calculate_financial_impact(timeline, performance_metrics, risk_assessment)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                risk_assessment, safety_violations, performance_metrics
            )
            
            # Calculate confidence in simulation
            confidence = await self._calculate_simulation_confidence(
                intervention_plan, timeline
            )
            
            # Create final result
            result = SimulationResult(
                simulation_id=simulation_id,
                initial_state=current_state,
                final_state=timeline[-1] if timeline else current_state,
                timeline=timeline,
                risk_assessment=risk_assessment,
                safety_violations=safety_violations,
                performance_metrics=performance_metrics,
                financial_impact=financial_impact,
                recommendations=recommendations,
                confidence=confidence
            )
            
            # Add to history
            self.simulation_history.append(result)
            
            # Maintain history size
            if len(self.simulation_history) > self.max_history_size:
                self.simulation_history = self.simulation_history[-self.max_history_size:]
            
            logger.info(f"🎮 Simulation completed: {simulation_id} "
                       f"(confidence: {confidence:.2f}, violations: {len(safety_violations)})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Simulation failed: {str(e)}")
            # Return failed simulation result
            return SimulationResult(
                simulation_id=f"failed_{intervention_plan.action_id}",
                initial_state=await self._get_current_system_state(),
                final_state=await self._get_current_system_state(),
                timeline=[],
                risk_assessment={'error': str(e)},
                safety_violations=['simulation_failed'],
                performance_metrics={},
                financial_impact={'total_cost': 0.0, 'error': 'Simulation failed'},
                recommendations=['Simulation failed - manual review required'],
                confidence=0.0
            )
    
    async def _get_current_system_state(self) -> SimulationState:
        """Get the current system state for simulation."""
        # In a real implementation, this would query actual system metrics
        # For now, we'll create a representative state
        
        variables = {}
        metrics = {}
        
        # Populate from system models
        for model_name, model_vars in self.system_models.items():
            for var_name, var_value in model_vars.items():
                full_var_name = f"{model_name}.{var_name}"
                variables[full_var_name] = var_value
                
                # Some variables become metrics
                if var_name in ['cpu_usage', 'memory_usage', 'disk_usage', 'error_rate']:
                    metrics[var_name] = var_value
        
        return SimulationState(
            timestamp=datetime.utcnow(),
            variables=variables,
            metrics=metrics,
            constraints=self.safety_constraints.copy()
        )
    
    async def _apply_intervention_effects(
        self, 
        state: SimulationState, 
        intervention: InterventionPlan
    ) -> SimulationState:
        """Apply the expected effects of an intervention to the system state."""
        modified_state = copy.deepcopy(state)
        
        # Apply expected effects from intervention plan
        for var_name, effect in intervention.expected_effects.items():
            if var_name in modified_state.variables:
                current_value = modified_state.variables[var_name]
                
                # Apply effect (could be additive, multiplicative, or absolute)
                if isinstance(effect, dict):
                    effect_type = effect.get('type', 'additive')
                    effect_value = effect.get('value', 0.0)
                    
                    if effect_type == 'additive':
                        modified_state.variables[var_name] = current_value + effect_value
                    elif effect_type == 'multiplicative':
                        modified_state.variables[var_name] = current_value * effect_value
                    elif effect_type == 'absolute':
                        modified_state.variables[var_name] = effect_value
                else:
                    # Direct value assignment
                    modified_state.variables[var_name] = effect
        
        # Update timestamp
        modified_state.timestamp = datetime.utcnow()
        
        return modified_state
    
    async def _run_temporal_simulation(
        self, 
        initial_state: SimulationState, 
        duration_seconds: float
    ) -> List[SimulationState]:
        """Run simulation over time using physical laws."""
        timeline = [initial_state]
        current_state = copy.deepcopy(initial_state)
        
        # Calculate number of steps
        num_steps = int(duration_seconds / self.simulation_granularity)
        
        for step in range(num_steps):
            # Apply physical laws to evolve system state
            evolved_state = await self._evolve_system_state(current_state)
            
            # Add to timeline
            timeline.append(evolved_state)
            current_state = evolved_state
        
        return timeline
    
    async def _evolve_system_state(self, current_state: SimulationState) -> SimulationState:
        """Evolve system state according to physical laws."""
        evolved_state = copy.deepcopy(current_state)
        evolved_state.timestamp += timedelta(seconds=self.simulation_granularity)
        
        # Apply CPU dynamics
        if 'server_infrastructure.cpu_usage' in evolved_state.variables:
            cpu_usage = evolved_state.variables['server_infrastructure.cpu_usage']
            
            # Apply cooling if CPU is not under load
            if cpu_usage < 0.3:
                cpu_usage = max(0.0, cpu_usage - self.physical_laws['cpu_dynamics']['cooling_rate'])
            else:
                # Apply heating under load
                cpu_usage = min(1.0, cpu_usage + self.physical_laws['cpu_dynamics']['utilization_rate'])
            
            evolved_state.variables['server_infrastructure.cpu_usage'] = cpu_usage
            
            # Update temperature based on CPU usage
            temp_key = 'server_infrastructure.temperature'
            if temp_key in evolved_state.variables:
                current_temp = evolved_state.variables[temp_key]
                heat_generation = cpu_usage * 20.0  # Degrees increase per unit CPU
                cooling = 2.0  # Natural cooling
                new_temp = current_temp + heat_generation - cooling
                evolved_state.variables[temp_key] = max(20.0, min(80.0, new_temp))
        
        # Apply memory dynamics
        if 'server_infrastructure.memory_usage' in evolved_state.variables:
            mem_usage = evolved_state.variables['server_infrastructure.memory_usage']
            
            # Apply garbage collection effect
            gc_effect = (1.0 - self.physical_laws['memory_dynamics']['garbage_collection_efficiency']) * 0.01
            mem_usage = max(0.0, mem_usage - gc_effect)
            
            # Apply fragmentation growth
            frag_growth = self.physical_laws['memory_dynamics']['fragmentation_growth']
            mem_usage = min(1.0, mem_usage + frag_growth)
            
            evolved_state.variables['server_infrastructure.memory_usage'] = mem_usage
        
        # Apply network dynamics
        if 'network.bandwidth_utilization' in evolved_state.variables:
            bandwidth_util = evolved_state.variables['network.bandwidth_utilization']
            
            # Apply congestion effects
            if bandwidth_util > self.physical_laws['network_dynamics']['bandwidth_saturation']:
                # Increase latency under congestion
                latency_key = 'network.latency'
                if latency_key in evolved_state.variables:
                    current_latency = evolved_state.variables[latency_key]
                    latency_increase = current_latency * self.physical_laws['network_dynamics']['latency_increase_rate']
                    evolved_state.variables[latency_key] = current_latency + latency_increase
            
            evolved_state.variables['network.bandwidth_utilization'] = min(1.0, bandwidth_util)
        
        # Apply database dynamics
        if 'database.query_rate' in evolved_state.variables:
            query_rate = evolved_state.variables['database.query_rate']
            
            # Performance degrades with higher query rates
            perf_scaling = self.physical_laws['database_dynamics']['query_performance_scaling']
            if query_rate > 100:  # High query threshold
                degradation = (query_rate - 100) * perf_scaling * 0.01
                evolved_state.variables['database.query_rate'] = max(0.0, query_rate + degradation)
        
        # Update metrics
        evolved_state.metrics = await self._extract_metrics(evolved_state.variables)
        
        return evolved_state
    
    async def _extract_metrics(self, variables: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics from system variables."""
        metrics = {}
        
        # CPU utilization
        if 'server_infrastructure.cpu_usage' in variables:
            metrics['cpu_usage'] = variables['server_infrastructure.cpu_usage']
        
        # Memory utilization
        if 'server_infrastructure.memory_usage' in variables:
            metrics['memory_usage'] = variables['server_infrastructure.memory_usage']
        
        # Disk usage
        if 'server_infrastructure.disk_usage' in variables:
            metrics['disk_usage'] = variables['server_infrastructure.disk_usage']
        
        # Error rate estimation
        cpu_usage = variables.get('server_infrastructure.cpu_usage', 0.0)
        memory_usage = variables.get('server_infrastructure.memory_usage', 0.0)
        
        # Higher resource usage correlates with higher error rates
        error_rate = 0.01 * (cpu_usage + memory_usage)  # Base error rate
        metrics['error_rate'] = min(1.0, error_rate)
        
        # Availability calculation
        metrics['availability'] = max(0.0, 1.0 - metrics['error_rate'])
        
        return metrics
    
    async def _assess_risks(self, timeline: List[SimulationState]) -> Dict[str, Any]:
        """Assess risks based on simulation timeline."""
        if not timeline:
            return {'risk_level': 'unknown'}
        
        risk_factors = []
        risk_scores = []
        
        for state in timeline:
            # CPU risk
            if state.metrics.get('cpu_usage', 0.0) > 0.8:
                risk_factors.append('high_cpu_utilization')
                risk_scores.append(state.metrics['cpu_usage'])
            
            # Memory risk
            if state.metrics.get('memory_usage', 0.0) > 0.75:
                risk_factors.append('high_memory_utilization')
                risk_scores.append(state.metrics['memory_usage'])
            
            # Error rate risk
            if state.metrics.get('error_rate', 0.0) > 0.03:
                risk_factors.append('high_error_rate')
                risk_scores.append(state.metrics['error_rate'] * 10)  # Scale for comparison
        
        # Calculate overall risk
        if risk_scores:
            avg_risk = np.mean(risk_scores)
            max_risk = np.max(risk_scores)
            
            risk_level = 'low' if avg_risk < 0.3 else 'medium' if avg_risk < 0.6 else 'high'
        else:
            avg_risk = 0.0
            max_risk = 0.0
            risk_level = 'low'
        
        return {
            'risk_level': risk_level,
            'average_risk_score': float(avg_risk),
            'maximum_risk_score': float(max_risk),
            'identified_risks': list(set(risk_factors)),
            'risk_timeline': len(risk_factors)
        }
    
    async def _check_safety_violations(self, timeline: List[SimulationState]) -> List[str]:
        """Check for safety constraint violations."""
        violations = []
        
        for state in timeline:
            # CPU utilization safety check
            if state.metrics.get('cpu_usage', 0.0) > self.safety_constraints['cpu_max']:
                violations.append('cpu_utilization_exceeded')
            
            # Memory utilization safety check
            if state.metrics.get('memory_usage', 0.0) > self.safety_constraints['memory_max']:
                violations.append('memory_utilization_exceeded')
            
            # Error rate safety check
            if state.metrics.get('error_rate', 0.0) > self.safety_constraints['error_rate_max']:
                violations.append('error_rate_exceeded')
            
            # Availability safety check
            if state.metrics.get('availability', 1.0) < self.safety_constraints['availability_min']:
                violations.append('availability_below_minimum')
        
        return list(set(violations))  # Remove duplicates
    
    async def _calculate_performance_metrics(self, timeline: List[SimulationState]) -> Dict[str, float]:
        """Calculate overall performance metrics from simulation."""
        if not timeline:
            return {}
        
        # Extract metric timelines
        cpu_usage_timeline = [state.metrics.get('cpu_usage', 0.0) for state in timeline]
        memory_usage_timeline = [state.metrics.get('memory_usage', 0.0) for state in timeline]
        error_rate_timeline = [state.metrics.get('error_rate', 0.0) for state in timeline]
        availability_timeline = [state.metrics.get('availability', 1.0) for state in timeline]
        
        return {
            'peak_cpu_usage': float(np.max(cpu_usage_timeline)),
            'average_cpu_usage': float(np.mean(cpu_usage_timeline)),
            'peak_memory_usage': float(np.max(memory_usage_timeline)),
            'average_memory_usage': float(np.mean(memory_usage_timeline)),
            'peak_error_rate': float(np.max(error_rate_timeline)),
            'average_error_rate': float(np.mean(error_rate_timeline)),
            'minimum_availability': float(np.min(availability_timeline)),
            'average_availability': float(np.mean(availability_timeline)),
            'stability_score': float(1.0 - np.std(availability_timeline))  # Higher = more stable
        }
    
    async def _calculate_financial_impact(
        self, 
        timeline: List[SimulationState], 
        metrics: Dict[str, float],
        risks: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate the financial 'Blast Radius' of the simulated intervention.
        Estimates costs based on resource usage and potential downtime.
        """
        # Base costs (hourly rates)
        COMPUTE_COST_PER_CORE_HOUR = 0.50
        MEMORY_COST_PER_GB_HOUR = 0.10
        STORAGE_COST_PER_GB_HOUR = 0.05
        DOWNTIME_COST_PER_SECOND = 1000.0  # Enterprise grade SLA breach penalty
        
        duration_hours = len(timeline) * self.simulation_granularity / 3600.0
        
        # Resource costs
        compute_cost = metrics.get('average_cpu_usage', 0) * 8 * COMPUTE_COST_PER_CORE_HOUR * duration_hours
        memory_cost = metrics.get('average_memory_usage', 0) * 32 * MEMORY_COST_PER_GB_HOUR * duration_hours
        
        # Risk-adjusted operational costs
        operational_risk_premium = risks.get('average_risk_score', 0) * 100.0
        
        # Downtime costs
        min_availability = metrics.get('minimum_availability', 1.0)
        downtime_seconds = (1.0 - min_availability) * len(timeline) * self.simulation_granularity
        downtime_cost = downtime_seconds * DOWNTIME_COST_PER_SECOND
        
        total_cost = compute_cost + memory_cost + operational_risk_premium + downtime_cost
        
        return {
            'compute_cost': round(compute_cost, 2),
            'downtime_risk_cost': round(downtime_cost, 2),
            'risk_premium': round(operational_risk_premium, 2),
            'total_estimated_impact': round(total_cost, 2)
        }
    
    async def _generate_recommendations(
        self, 
        risk_assessment: Dict[str, Any], 
        safety_violations: List[str], 
        performance_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on simulation results."""
        recommendations = []
        
        # Risk-based recommendations
        risk_level = risk_assessment.get('risk_level', 'low')
        if risk_level == 'high':
            recommendations.append('Consider reducing intervention scope or adding safeguards')
            recommendations.append('Implement gradual rollout with monitoring')
        elif risk_level == 'medium':
            recommendations.append('Monitor system closely during implementation')
        
        # Safety violation recommendations
        if 'cpu_utilization_exceeded' in safety_violations:
            recommendations.append('Add CPU monitoring and auto-scaling')
        if 'memory_utilization_exceeded' in safety_violations:
            recommendations.append('Implement memory usage alerts and optimization')
        if 'error_rate_exceeded' in safety_violations:
            recommendations.append('Add error rate monitoring and circuit breakers')
        
        # Performance recommendations
        if performance_metrics.get('average_cpu_usage', 0.0) > 0.7:
            recommendations.append('Consider off-peak scheduling for resource-intensive operations')
        if performance_metrics.get('stability_score', 1.0) < 0.8:
            recommendations.append('Implement additional resilience measures')
        
        if not recommendations:
            recommendations.append('Intervention appears safe to proceed with standard monitoring')
        
        return recommendations
    
    async def _calculate_simulation_confidence(
        self, 
        intervention: InterventionPlan, 
        timeline: List[SimulationState]
    ) -> float:
        """Calculate confidence level in the simulation results."""
        confidence_factors = []
        
        # Complexity factor - simpler interventions have higher confidence
        complexity_score = len(intervention.target_variables) / 10.0
        confidence_factors.append(1.0 - min(0.5, complexity_score))
        
        # Duration factor - shorter simulations have higher confidence
        duration_score = min(1.0, intervention.duration_seconds / 3600.0)  # Normalize to hours
        confidence_factors.append(1.0 - min(0.3, duration_score * 0.5))
        
        # Stability factor - stable baselines increase confidence
        if timeline:
            availability_timeline = [state.metrics.get('availability', 1.0) for state in timeline]
            stability = 1.0 - np.std(availability_timeline)
            confidence_factors.append(stability)
        
        # Return geometric mean of confidence factors
        if confidence_factors:
            confidence = np.prod(confidence_factors) ** (1/len(confidence_factors))
            return max(0.1, min(1.0, confidence))  # Clamp between 0.1 and 1.0
        
        return 0.5  # Default confidence
    
    def get_simulation_history(self, limit: int = 10) -> List[SimulationResult]:
        """Get recent simulation history."""
        return self.simulation_history[-limit:] if self.simulation_history else []
    
    def get_system_model_status(self) -> Dict[str, Any]:
        """Get status of system models."""
        return {
            'models_loaded': list(self.system_models.keys()),
            'physical_laws_loaded': list(self.physical_laws.keys()),
            'safety_constraints_defined': len(self.safety_constraints),
            'simulation_history_size': len(self.simulation_history)
        }

    async def inject_noise(self, mode: str = "drift") -> bool:
        """
        Inject "Gremlin" noise into system models for chaos testing.
        
        Args:
            mode: Type of noise to inject ("drift", "spike", "freeze")
            
        Returns:
            bool: Success status
        """
        logger.warning(f"👺 GREMLIN MODE ACTIVATED: Injecting '{mode}' noise into digital twin")
        
        try:
            if mode == "drift":
                # Gradually drift values over time (handled by physical laws modifiers)
                # For now, we'll immediately skew some baselines
                self.system_models['server_infrastructure']['cpu_usage'] += 0.15
                self.system_models['network']['latency'] *= 1.5
                
            elif mode == "spike":
                # Immediate massive spike in resource usage
                self.system_models['server_infrastructure']['cpu_usage'] = 0.99
                self.system_models['server_infrastructure']['memory_usage'] = 0.95
                self.system_models['database']['query_rate'] *= 5.0
                
            elif mode == "freeze":
                # System becomes unresponsive
                self.system_models['application']['response_time'] = 30.0  # 30s timeout
                self.system_models['application']['availability'] = 0.0
                
            else:
                logger.error(f"❌ Unknown Gremlin mode: {mode}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to inject Gremlin noise: {str(e)}")
            return False

# Global instance
digital_twin_simulator: Optional[DigitalTwinSimulator] = None

async def initialize_digital_twin_simulator(simulation_granularity: float = 1.0) -> bool:
    """
    Initialize the global digital twin simulator instance.
    
    Args:
        simulation_granularity: Time step for simulations in seconds
        
    Returns:
        bool: Success status
    """
    global digital_twin_simulator
    
    try:
        digital_twin_simulator = DigitalTwinSimulator(simulation_granularity=simulation_granularity)
        return await digital_twin_simulator.initialize()
    except Exception as e:
        logger.error(f"❌ Failed to initialize digital twin simulator: {str(e)}")
        return False

def get_digital_twin_simulator() -> DigitalTwinSimulator:
    """Get the global digital twin simulator instance."""
    if digital_twin_simulator is None:
        raise RuntimeError("Digital twin simulator not initialized")
    return digital_twin_simulator