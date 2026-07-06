use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
    process::Command,
};

struct Project {
    name: String,
    description: String,
    dir: PathBuf,
}

fn parse_field<'a>(toml: &'a str, field: &str) -> Option<&'a str> {
    toml.lines()
        .find(|l| l.starts_with(field))
        .and_then(|l| l.splitn(2, '=').nth(1))
        .map(|v| v.trim().trim_matches('"'))
}

fn discover_projects(workspace_root: &PathBuf) -> Vec<Project> {
    let projects_dir = workspace_root.join("projects");

    let Ok(entries) = fs::read_dir(&projects_dir) else {
        return vec![];
    };

    let mut projects: Vec<_> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| {
            let cargo_toml = e.path().join("Cargo.toml");
            let contents = fs::read_to_string(&cargo_toml).ok()?;
            let name = parse_field(&contents, "name")?.to_string();
            let description = parse_field(&contents, "description")
                .unwrap_or("sem descrição")
                .to_string();
            Some(Project { name, description, dir: e.path() })
        })
        .collect();

    projects.sort_by(|a, b| a.dir.cmp(&b.dir));
    projects
}

fn main() {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let projects = discover_projects(&workspace_root);

    if projects.is_empty() {
        eprintln!("Nenhum projeto encontrado em projects/");
        return;
    }

    println!("\nia-explore\n");
    for (i, p) in projects.iter().enumerate() {
        println!("  {}. {} — {}", i + 1, p.name, p.description);
    }
    println!();
    print!("Escolha: ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();

    let choice: usize = match input.trim().parse::<usize>() {
        Ok(n) if n >= 1 && n <= projects.len() => n,
        _ => {
            eprintln!("Opção inválida.");
            return;
        }
    };

    let project = &projects[choice - 1];
    println!("\nExecutando {}...\n", project.name);

    Command::new("cargo")
        .args(["run", "-p", &project.name])
        .current_dir(&workspace_root)
        .status()
        .unwrap();
}
