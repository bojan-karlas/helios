# Bash reads this instead of ~/.bashrc for the project subshell only.
if [[ -f "$HOME/.bashrc" ]]; then
    source "$HOME/.bashrc" || exit
fi

helios_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || exit
export VIRTUAL_ENV_DISABLE_PROMPT=1
source "$helios_root/.venv/bin/activate" || exit
PS1="(helios) ${PS1:-\\u@\\h:\\w\\$ }"
unset helios_root
