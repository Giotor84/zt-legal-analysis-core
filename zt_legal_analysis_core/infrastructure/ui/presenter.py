# -*- coding: utf-8 -*-
# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
Presenter - View layer per UI Rich/Console.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from rich.console import Console
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table


class Presenter:
    def __init__(self, console: Console):
        self.console = console
        self._batch_live: Optional[Live] = None
        self._batch_progress: Optional[Progress] = None
        self._batch_task_id: Optional[int] = None
        self._batch_state: Dict[str, Any] = {}

    # --- Base output helpers ---
    def text(self, *args: Any, sep: str = " ", end: str = "\n", style: Optional[str] = None) -> None:
        message = sep.join(str(a) for a in args)
        self.console.print(message, style=style, end=end)

    def info(self, message: str) -> None:
        self.console.print(message)

    def success(self, message: str) -> None:
        self.console.print(message, style="green")

    def warning(self, message: str) -> None:
        self.console.print(message, style="yellow")

    def error(self, message: str) -> None:
        self.console.print(message, style="red")

    # --- Banners ---
    def show_banner(self, title: str, subtitle: str, icon: str = "🏛️", style: str = "bold blue") -> None:
        try:
            banner = (
                "\n"
                "╔═══════════════════════════════════════════════════════════╗\n"
                f"║{(' ' * 59)}║\n"
                f"║     {icon}  {title:<45}║\n"
                f"║{(' ' * 59)}║\n"
                f"║     {subtitle:<49}║\n"
                f"║{(' ' * 59)}║\n"
                "╚═══════════════════════════════════════════════════════════╝\n"
            )
            self.console.print(banner, style=style)
        except UnicodeEncodeError:
            # Fallback ASCII per Windows console
            banner_ascii = (
                "\n"
                "============================================================\n"
                f"  {title}\n"
                f"  {subtitle}\n"
                "============================================================\n"
            )
            self.console.print(banner_ascii, style=style)

    # --- Tables & Panels ---
    def show_status_table(self, rows: List[Dict[str, str]]) -> None:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Componente", style="white")
        table.add_column("Status", justify="center")
        table.add_column("Dettagli", style="dim")
        for r in rows:
            table.add_row(r.get("component", ""), r.get("status", ""), r.get("details", ""))
        self.console.print(table)

    def show_panel(self, title: str, body: str, style: Optional[str] = None) -> None:
        self.console.print(Panel.fit(body, title=title, style=style))

    # --- Batch Live Dashboard ---
    def start_batch_live(self, total_files: int) -> None:
        self._batch_state = {
            "current_file": "-",
            "vram_used": "-",
            "completed": 0,
            "errors": 0,
            "total": total_files,
        }
        self._batch_progress = Progress(
            TextColumn("[bold]Batch[/bold]"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(compact=True),
            console=self.console,
        )
        self._batch_task_id = self._batch_progress.add_task("", total=total_files)
        self._batch_live = Live(self._render_batch_dashboard(), console=self.console, refresh_per_second=8)
        self._batch_live.start()

    def update_batch_live(
        self,
        current_file: Optional[str] = None,
        completed: Optional[int] = None,
        errors: Optional[int] = None,
        vram_used: Optional[str] = None,
    ) -> None:
        if not self._batch_live or not self._batch_progress or self._batch_task_id is None:
            return

        if current_file is not None:
            self._batch_state["current_file"] = current_file
        if completed is not None:
            self._batch_state["completed"] = completed
            self._batch_progress.update(self._batch_task_id, completed=completed)
        if errors is not None:
            self._batch_state["errors"] = errors
        if vram_used is not None:
            self._batch_state["vram_used"] = vram_used

        self._batch_live.update(self._render_batch_dashboard())

    def stop_batch_live(self) -> None:
        if self._batch_live:
            self._batch_live.stop()
        self._batch_live = None
        self._batch_progress = None
        self._batch_task_id = None
        self._batch_state = {}

    def log_file_complete(
        self,
        filename: str,
        elapsed: float,
        ok: bool = True,
        index: Optional[int] = None,
        total: Optional[int] = None,
        vram_used: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> None:
        status = "[green]OK[/green]" if ok else "[red]ERR[/red]"
        position = f"{index}/{total}" if index is not None and total is not None else "-"
        vram = vram_used or "-"
        step = stage or "Analisi Finanziaria..."
        self.console.print(f"{status} File {position} | VRAM: {vram} | Stato: {step}")

    def _render_batch_dashboard(self) -> Group:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("File Corrente", style="white")
        table.add_column("VRAM Occupata", justify="center")
        table.add_column("Documenti Completati", justify="center")
        table.add_column("Errori", justify="center")

        table.add_row(
            str(self._batch_state.get("current_file", "-")),
            str(self._batch_state.get("vram_used", "-")),
            f"{self._batch_state.get('completed', 0)}/{self._batch_state.get('total', 0)}",
            str(self._batch_state.get("errors", 0)),
        )

        return Group(table, self._batch_progress or "")

    def show_criticalities(self, criticalities: List[Dict[str, Any]]) -> None:
        table = Table(title="Criticità Rilevate", show_header=True)
        table.add_column("Codice", style="cyan")
        table.add_column("Titolo")
        table.add_column("Livello", justify="center")
        for c in criticalities:
            level = c.get("level", "unknown")
            level_style = "red bold" if level == "blocking" else "yellow" if level == "high" else "white"
            table.add_row(
                c.get("code", ""),
                c.get("title", ""),
                f"[{level_style}]{level.upper()}[/{level_style}]",
            )
        self.console.print(table)

    def show_batch_summary(self, results: Iterable[Any]) -> None:
        summary_table = Table(title="📊 RIEPILOGO BATCH", show_header=True)
        summary_table.add_column("Fascicolo", style="cyan")
        summary_table.add_column("Status", justify="center")
        summary_table.add_column("Score", justify="center")
        summary_table.add_column("Rating", justify="center")
        summary_table.add_column("Azionabile", justify="center")

        successful = 0
        total = 0
        for r in results:
            total += 1
            status = "✅" if r.status.value == "completed" else "❌"
            if r.status.value == "completed":
                successful += 1
            score = r.scoring.get("score_finale", "-") if r.scoring else "-"
            rating = r.scoring.get("rating", "-") if r.scoring else "-"
            actionable = "SÌ" if r.scoring and r.scoring.get("is_actionable") else "NO"
            summary_table.add_row(r.fascicolo_id, status, str(score), rating, actionable)

        self.console.print(summary_table)
        self.console.print(f"\n[bold]Totale:[/bold] {total} elaborati, {successful} successo, {total-successful} falliti")

    def show_agent_stats(self, by_agent: Dict[str, Dict[str, int]]) -> None:
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Agent", style="cyan")
        table.add_column("Esecuzioni", justify="right")
        table.add_column("Successi", justify="right", style="green")
        table.add_column("Errori", justify="right", style="red")
        for agent_name, agent_stats in by_agent.items():
            total = agent_stats.get("total", 0)
            if total > 0:
                table.add_row(
                    agent_name,
                    str(total),
                    str(agent_stats.get("success", 0)),
                    str(agent_stats.get("failed", 0)),
                )
        self.console.print(table)

    # --- Checkpoint / Agents ---
    def show_checkpoint_list(self, rows: List[Dict[str, str]]) -> None:
        table = Table(title="📍 Checkpoint Incompleti")
        table.add_column("Fascicolo ID", style="cyan")
        table.add_column("Progresso", style="yellow")
        table.add_column("Ultimo Step", style="magenta")
        table.add_column("Timestamp")
        for r in rows:
            table.add_row(r["fascicolo_id"], r["percentage"], r["last_step"], r["last_checkpoint"])
        self.console.print(table)

    def show_checkpoint_progress(self, fascicolo_id: str, progress: Dict[str, Any]) -> None:
        body = (
            f"[bold]Fascicolo:[/bold] {fascicolo_id}\n"
            f"[bold]Progresso:[/bold] {progress['percentage']}%\n"
            f"[bold]Step completati:[/bold] {progress['completed_steps']}/{progress['total_steps']}\n"
            f"[bold]Ultimo step:[/bold] {progress['last_step']}\n"
            f"[bold]Timestamp:[/bold] {progress['last_checkpoint']}\n\n"
            f"[bold]Step rimanenti:[/bold]\n" + "\n".join(f"  • {s}" for s in progress["remaining_steps"])
        )
        self.show_panel("📊 Stato Elaborazione", body)

    def show_agents_status(self, stats: Dict[str, Any], active_agents: Dict[str, List[str]]) -> None:
        total_exec = stats.get("total_executions", 0)
        total_success = stats.get("total_success", 0)
        total_failed = stats.get("total_failed", 0)
        success_rate = (total_success / total_exec * 100) if total_exec > 0 else 0

        body = (
            "[bold cyan]📊 AGENT COORDINATOR STATUS[/bold cyan]\n\n"
            f"[bold]Esecuzioni Totali:[/bold] {total_exec}\n"
            f"[bold]Successi:[/bold] {total_success} ({success_rate:.1f}%)\n"
            f"[bold]Fallimenti:[/bold] {total_failed}\n"
            f"[bold]Agenti Attivi:[/bold] {stats.get('active_count', 0)}"
        )
        self.show_panel("🎯 Agent Coordinator", body)

        table = Table(title="🤖 Stato Agenti")
        table.add_column("Agent", style="cyan")
        table.add_column("Attivi", justify="center", style="yellow")
        table.add_column("Totali", justify="center")
        table.add_column("Successi", justify="center", style="green")
        table.add_column("Fallimenti", justify="center", style="red")
        table.add_column("Success Rate", justify="center")

        for agent_name in stats.get("by_agent", {}).keys():
            agent_stats = stats.get("by_agent", {}).get(agent_name, {})
            active = stats.get("active_agents", {}).get(agent_name, 0)
            total = agent_stats.get("total", 0)
            success = agent_stats.get("success", 0)
            failed = agent_stats.get("failed", 0)
            rate = (success / total * 100) if total > 0 else 0
            status_emoji = "🟢" if active == 0 else "🔵"
            table.add_row(
                f"{status_emoji} {agent_name}",
                str(active) if active > 0 else "-",
                str(total),
                str(success),
                str(failed),
                f"{rate:.1f}%" if total > 0 else "-",
            )
        self.console.print(table)

        if active_agents:
            self.console.print("\n[bold yellow]⚡ Agenti in Esecuzione:[/bold yellow]")
            for agent, fascicoli in active_agents.items():
                self.console.print(f"   • {agent}: {len(fascicoli)} fascicoli")
                for fasc in fascicoli[:3]:
                    self.console.print(f"      - {fasc}")
        else:
            self.console.print("\n[green]✅ Nessun agent attivo al momento[/green]")
