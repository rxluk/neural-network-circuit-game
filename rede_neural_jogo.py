"""
AI car learning to drive — entry point.
Run this file to start the simulation.
"""

from sim.visualizacao import SimuladorAprendizado

if __name__ == "__main__":
    simulador = SimuladorAprendizado()
    simulador.iniciar_animacao()
