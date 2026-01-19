"""
demo_interactive.py

Interface de démonstration interactive pour la vidéo.
Simule l'expérience d'un étudiant utilisant le système.
"""

import pandas as pd
import json
import time
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

from shared.state import create_initial_state
from orchestrator.orchestrator import Orchestrator
from config.settings import Config

console = Console()


def print_welcome():
    """Écran de bienvenue"""
    console.clear()
    
    welcome_text = """
# 🎓 Bienvenue sur EduLearn AI

## Votre Assistant d'Apprentissage Personnalisé

Ce système utilise l'intelligence artificielle pour :
- ✨ Analyser votre profil d'apprentissage
- 🗺️  Créer un parcours personnalisé
- 📚 Générer du contenu adapté à vous
- 💡 Recommander les meilleures ressources
- 🔍 Expliquer chaque décision de manière transparente

**Prêt à commencer votre parcours d'apprentissage personnalisé ?**
"""
    
    console.print(Panel(
        Markdown(welcome_text),
        title="🎓 EduLearn AI",
        border_style="bright_blue",
        padding=(1, 2)
    ))
    
    input("\n👉 Appuyez sur Entrée pour commencer...")


def student_questionnaire():
    """Questionnaire pour l'étudiant"""
    console.clear()
    console.print("\n[bold cyan]📝 Questionnaire Initial[/bold cyan]\n")
    
    console.print("Pour personnaliser votre expérience, répondez à ces questions :\n")
    
    # Question 1
    name = Prompt.ask("[yellow]1. Quel est votre prénom ?[/yellow]", default="Balkis")
    
    # Question 2
    console.print("\n[yellow]2. Quel est votre objectif d'apprentissage ?[/yellow]")
    console.print("   a) Apprendre les bases")
    console.print("   b) Améliorer mes compétences")
    console.print("   c) Maîtriser des concepts avancés")
    goal = Prompt.ask("Votre choix", choices=["a", "b", "c"], default="b")
    
    goal_map = {
        "a": "Je veux apprendre les fondamentaux",
        "b": "Je veux améliorer mes compétences existantes",
        "c": "Je veux maîtriser des concepts avancés"
    }
    
    # Question 3
    console.print("\n[yellow]3. Combien de temps pouvez-vous consacrer par session ?[/yellow]")
    console.print("   a) 15-30 minutes")
    console.print("   b) 30-60 minutes")
    console.print("   c) Plus d'une heure")
    time_commit = Prompt.ask("Votre choix", choices=["a", "b", "c"], default="b")
    
    # Question 4
    console.print("\n[yellow]4. Quel style d'apprentissage préférez-vous ?[/yellow]")
    console.print("   a) Visuel (diagrammes, vidéos)")
    console.print("   b) Pratique (exercices, projets)")
    console.print("   c) Théorique (lectures, concepts)")
    console.print("   d) Mixte")
    style = Prompt.ask("Votre choix", choices=["a", "b", "c", "d"], default="d")
    
    return {
        'name': name,
        'goal': goal_map[goal],
        'time_commitment': time_commit,
        'learning_style': style
    }


def show_profiling(state, student_info):
    """Affiche l'analyse du profil"""
    console.clear()
    
    console.print("\n[bold cyan]🔍 ÉTAPE 1/5 : Analyse de votre Profil[/bold cyan]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Analyse en cours...", total=100)
        
        for i in range(100):
            time.sleep(0.02)  # Ralentir pour l'effet visuel
            progress.update(task, advance=1)
    
    profile = state.get('profile', {})
    
    # Créer un tableau de profil
    table = Table(title="📊 Votre Profil d'Apprentissage", 
                  title_style="bold magenta",
                  show_header=True,
                  header_style="bold cyan")
    
    table.add_column("Caractéristique", style="cyan", width=25)
    table.add_column("Valeur", style="green", width=40)
    
    table.add_row("👤 Nom", student_info['name'])
    table.add_row("🎯 Objectif", student_info['goal'])
    table.add_row("📊 Score Actuel", f"{profile.get('avg_score', 0):.1f}/100")
    table.add_row("⚡ Niveau d'Engagement", profile.get('engagement_level', 'Medium'))
    table.add_row("🎨 Style d'Apprentissage", profile.get('learning_style', 'balanced_learner'))
    table.add_row("👥 Groupe de Profil", f"Cluster {profile.get('cluster_id', 0)}")
    
    console.print("\n")
    console.print(table)
    
    # Insights
    console.print("\n[bold green]✨ Insights de votre profil :[/bold green]")
    
    score = profile.get('avg_score', 0)
    if score >= 80:
        console.print("  • Vous avez un excellent niveau de compétence!")
    elif score >= 60:
        console.print("  • Vous avez de bonnes bases, continuez!")
    else:
        console.print("  • Vous êtes au début de votre parcours, c'est parfait!")
    
    console.print(f"  • Votre style {profile.get('learning_style', 'équilibré')} est idéal pour un apprentissage varié")
    console.print(f"  • Vous faites partie d'un groupe similaire d'apprenants")
    
    input("\n👉 Appuyez sur Entrée pour voir votre parcours personnalisé...")


def show_learning_path(state, student_info):
    """Affiche le parcours d'apprentissage"""
    console.clear()
    
    console.print("\n[bold cyan]🗺️  ÉTAPE 2/5 : Votre Parcours Personnalisé[/bold cyan]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Création de votre parcours optimal...", total=100)
        
        for i in range(100):
            time.sleep(0.03)
            progress.update(task, advance=1)
    
    learning_path = state.get('learning_path', [])
    
    console.print(f"\n[bold green]✅ Parcours créé pour {student_info['name']}![/bold green]\n")
    
    # Afficher le parcours comme une timeline
    for i, unit in enumerate(learning_path, 1):
        console.print(f"[bold cyan]{'─'*60}[/bold cyan]")
        console.print(f"[bold yellow]📍 Étape {i}: {unit['concept'].upper()}[/bold yellow]")
        console.print(f"[cyan]{'─'*60}[/cyan]")
        console.print(f"  📊 Niveau: [green]{unit['difficulty']}[/green]")
        console.print(f"  ⏱️  Durée estimée: [yellow]{unit['estimated_duration']} minutes[/yellow]")
        
        if unit.get('prerequisites'):
            console.print(f"  📋 Prérequis: {', '.join(unit['prerequisites'])}")
        
        console.print(f"  🎯 Objectifs:")
        for obj in unit.get('learning_objectives', [])[:2]:
            console.print(f"     • {obj}")
        console.print()
    
    console.print(f"[bold cyan]{'─'*60}[/bold cyan]")
    console.print(f"\n[bold green]📈 Temps total estimé: {sum(u['estimated_duration'] for u in learning_path)} minutes[/bold green]")
    
    input("\n👉 Appuyez sur Entrée pour voir le contenu généré...")


def show_generated_content(state, student_info):
    """Affiche le contenu généré - VERSION DÉTAILLÉE"""
    console.clear()
    
    console.print("\n[bold cyan]📚 ÉTAPE 3/5 : Contenu Généré pour Vous[/bold cyan]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Génération de contenu personnalisé avec RAG...", total=100)
        
        for i in range(100):
            time.sleep(0.025)
            progress.update(task, advance=1)
    
    generated_content = state.get('generated_content', [])
    
    if not generated_content:
        console.print("[yellow]⚠️  Aucun contenu généré[/yellow]")
        return
    
    # Afficher chaque contenu de manière détaillée
    for i, content in enumerate(generated_content, 1):
        console.print(f"\n[bold magenta]{'='*70}[/bold magenta]")
        console.print(f"[bold yellow]📖 Module {i}: {content['concept'].upper()}[/bold yellow]")
        console.print(f"[bold magenta]{'='*70}[/bold magenta]\n")
        
        # Explication
        console.print("[bold cyan]📝 Explication Personnalisée:[/bold cyan]")
        console.print(Panel(
            content['explanation'],
            border_style="cyan",
            padding=(1, 2)
        ))
        
        # Exemples
        if content.get('examples'):
            console.print("\n[bold green]💡 Exemples Pratiques:[/bold green]")
            for j, example in enumerate(content['examples'], 1):
                console.print(f"  {j}. {example}")
        
        # Quiz
        quiz_questions = content.get('quiz', [])
        if quiz_questions:
            console.print(f"\n[bold yellow]❓ Quiz Interactif ({len(quiz_questions)} question{'s' if len(quiz_questions) > 1 else ''}):[/bold yellow]")
            
            for q_num, question in enumerate(quiz_questions, 1):
                console.print(f"\n[cyan]Question {q_num}:[/cyan] {question['question']}")
                
                for opt_num, option in enumerate(question.get('options', []), 1):
                    console.print(f"  {opt_num}. {option}")
                
                # Simulation de réponse
                console.print(f"\n[dim]✓ Bonne réponse: Option {question.get('correct_answer', 1)}[/dim]")
                if question.get('explanation'):
                    console.print(f"[dim]📌 Explication: {question['explanation']}[/dim]")
        
        # Montrer les sources RAG
        if content.get('sources'):
            console.print(f"\n[dim]📚 Sources utilisées (RAG): {len(content['sources'])} documents[/dim]")
        
        if i < len(generated_content):
            input(f"\n👉 Appuyez sur Entrée pour voir le module suivant...")
    
    input("\n\n👉 Appuyez sur Entrée pour voir les recommandations...")


def show_recommendations(state, student_info):
    """Affiche les recommandations"""
    console.clear()
    
    console.print("\n[bold cyan]💡 ÉTAPE 4/5 : Recommandations Personnalisées[/bold cyan]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Analyse et recommandations...", total=100)
        
        for i in range(100):
            time.sleep(0.02)
            progress.update(task, advance=1)
    
    recommendations = state.get('recommendations', {})
    
    console.print(f"\n[bold green]✨ Recommandations pour {student_info['name']}:[/bold green]\n")
    
    # Primary recommendation
    primary = recommendations.get('primary', {})
    if primary:
        console.print(Panel(
            f"[bold yellow]🎯 Priorité Absolue[/bold yellow]\n\n"
            f"Commencez par: [cyan]{primary.get('concept', 'N/A')}[/cyan]\n"
            f"Confiance: [green]{primary.get('confidence', 0)*100:.0f}%[/green]",
            border_style="yellow",
            padding=(1, 2)
        ))
    
    # Next steps
    next_steps = recommendations.get('next_steps', [])
    if next_steps:
        console.print("\n[bold cyan]📋 Prochaines Étapes:[/bold cyan]")
        for i, step in enumerate(next_steps, 1):
            console.print(f"  {i}. {step}")
    
    # Similar learners
    if recommendations.get('similar_learners_preferences'):
        console.print("\n[bold magenta]👥 Ce que des apprenants similaires ont aimé:[/bold magenta]")
        for pref in recommendations.get('similar_learners_preferences', [])[:3]:
            console.print(f"  • {pref}")
    
    input("\n👉 Appuyez sur Entrée pour comprendre pourquoi ces recommandations...")


def show_explanations(state, student_info):
    """Affiche les explications (XAI)"""
    console.clear()
    
    console.print("\n[bold cyan]🔍 ÉTAPE 5/5 : Transparence & Explications[/bold cyan]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Génération des explications...", total=100)
        
        for i in range(100):
            time.sleep(0.02)
            progress.update(task, advance=1)
    
    explanations = state.get('explanations', {})
    
    # Vue apprenant
    learner_view = explanations.get('learner_view', '')
    if learner_view:
        console.print(Panel(
            Markdown(learner_view),
            title="💬 Pourquoi ces recommandations ?",
            border_style="green",
            padding=(1, 2)
        ))
    
    # Feature importance
    feature_importance = explanations.get('feature_importance', {})
    if feature_importance.get('top_influencers'):
        console.print("\n[bold cyan]📊 Facteurs Clés qui ont Influencé vos Recommandations:[/bold cyan]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Facteur", style="cyan", width=30)
        table.add_column("Impact", style="yellow", width=30)
        
        for factor, score in feature_importance['top_influencers']:
            bar = "█" * int(score * 20)
            table.add_row(factor, f"{bar} {score:.2%}")
        
        console.print(table)
    
    # Counterfactuals
    counterfactuals = explanations.get('counterfactuals', [])
    if counterfactuals:
        console.print("\n[bold yellow]💭 Comment Améliorer Votre Parcours:[/bold yellow]\n")
        for cf in counterfactuals[:3]:
            console.print(f"  • {cf}")
    
    input("\n👉 Appuyez sur Entrée pour voir le résumé final...")


def show_summary(state, student_info):
    """Résumé final"""
    console.clear()
    
    console.print("\n[bold magenta]🎉 RÉSUMÉ DE VOTRE SESSION[/bold magenta]\n")
    
    profile = state.get('profile', {})
    learning_path = state.get('learning_path', [])
    generated_content = state.get('generated_content', [])
    
    summary_text = f"""
# 📊 Votre Parcours Personnalisé est Prêt!

## 👤 Profil
- **Nom**: {student_info['name']}
- **Score**: {profile.get('avg_score', 0):.1f}/100
- **Style**: {profile.get('learning_style', 'balanced')}
- **Engagement**: {profile.get('engagement_level', 'Medium')}

## 🗺️ Parcours
- **{len(learning_path)} modules** créés spécialement pour vous
- **Durée totale**: {sum(u['estimated_duration'] for u in learning_path)} minutes
- **Niveau**: {learning_path[0]['difficulty'] if learning_path else 'N/A'}

## 📚 Contenu Généré
- **{len(generated_content)} ressources** personnalisées
- **{sum(len(c.get('quiz', [])) for c in generated_content)} questions** de quiz
- Généré avec RAG pour garantir la qualité

## 💡 Prochaines Étapes
1. Commencez par le premier module
2. Complétez les quiz pour valider
3. Le système s'adaptera à vos progrès

---

**🚀 Prêt à apprendre de manière intelligente ?**
"""
    
    console.print(Panel(
        Markdown(summary_text),
        border_style="bright_magenta",
        padding=(1, 2)
    ))
    
    # Option de sauvegarder
    save = Confirm.ask("\n💾 Voulez-vous sauvegarder ces résultats ?")
    
    if save:
        output_file = f"outputs/session_{student_info['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_data = {
            'student_info': student_info,
            'profile': profile,
            'learning_path': learning_path,
            'generated_content': generated_content,
            'recommendations': state.get('recommendations', {}),
            'explanations': state.get('explanations', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"\n[green]✅ Résultats sauvegardés dans:[/green] [cyan]{output_file}[/cyan]")
    
    console.print("\n[bold green]Merci d'avoir utilisé EduLearn AI! 🎓[/bold green]")


def main():
    """Fonction principale de la démo interactive"""
    
    # 1. Écran de bienvenue
    print_welcome()
    
    # 2. Questionnaire étudiant
    student_info = student_questionnaire()
    
    console.print(f"\n[bold green]✨ Parfait {student_info['name']}! Lançons votre analyse...[/bold green]")
    time.sleep(2)
    
    # 3. Charger les données et créer l'état initial
    console.clear()
    console.print("\n[cyan]📂 Chargement de vos données...[/cyan]\n")
    
    df = pd.read_csv(Config.system.data_path)
    sample_learner = df.iloc[0].to_dict()
    learner_id = f"{sample_learner['id_student']}_{sample_learner['code_module']}"
    
    # Enrichir avec les réponses du questionnaire
    sample_learner['name'] = student_info['name']
    sample_learner['goal'] = student_info['goal']
    
    initial_state = create_initial_state(learner_id, sample_learner)
    
    # 4. Créer l'orchestrator et exécuter (en silence)
    console.print("[cyan]🤖 Initialisation du système multi-agent...[/cyan]\n")
    
    # Rediriger temporairement les prints
    import sys
    import io
    
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    orchestrator = Orchestrator()
    final_state = orchestrator.run(initial_state)
    
    sys.stdout = old_stdout
    
    # 5. Afficher les résultats étape par étape
    show_profiling(final_state, student_info)
    show_learning_path(final_state, student_info)
    show_generated_content(final_state, student_info)
    show_recommendations(final_state, student_info)
    show_explanations(final_state, student_info)
    show_summary(final_state, student_info)


if __name__ == "__main__":
    main()
