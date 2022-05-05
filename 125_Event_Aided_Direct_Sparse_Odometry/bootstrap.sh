#! /bin/sh

CONF_URL=${CONF_URL:=https://github.com/uzh-rpg/eds-buildconf.git}
RUBY=ruby
AUTOPROJ_BOOTSTRAP_URL=http://rock-robotics.org/master/autoproj_bootstrap
BOOTSTRAP_ARGS=

if test -n "$1" && test "$1" != "dev" && test "$1" != "localdev"; then
    RUBY=$1
    shift 1
    RUBY_USER_SELECTED=1
fi

set -e

if ! which $RUBY > /dev/null 2>&1; then
    echo "cannot find the ruby executable. On Ubuntu 16.04 and above, you should run"
    echo "  sudo apt-get install ruby"
    echo "or on Ubuntu 14.04"
    echo "  sudo apt-get install ruby2.0"
    exit 1
fi

RUBY_VERSION_VALID=`$RUBY -e 'STDOUT.puts RUBY_VERSION.to_f >= 2.0'`

if [ "x$RUBY_VERSION_VALID" != "xtrue" ]; then
    if test "$RUBY_USER_SELECTED" = "1"; then
        echo "You selected $RUBY as the ruby executable, and it is not providing Ruby >= 2.0"
    else
        cat <<EOMSG
ruby --version reports
  `$RUBY --version`
The supported version for Rock is ruby >= 2.0. I don't know if you have it
installed, and if you do what name it has. You will have to select a Ruby
executable by passing it on the command line, as e.g.
  sh bootstrap.sh ruby2.1
EOMSG
        exit 1
    fi
fi

if ! test -f $PWD/autoproj_bootstrap; then
    if which wget > /dev/null; then
        DOWNLOADER=wget
    elif which curl > /dev/null; then
        DOWNLOADER=curl
    else
        echo "I can find neither curl nor wget, either install one of these or"
        echo "download the following script yourself, and re-run this script"
        exit 1
    fi
    $DOWNLOADER $AUTOPROJ_BOOTSTRAP_URL
fi

CONF_URL=${CONF_URL#*//}
CONF_SITE=${CONF_URL%%/*}
CONF_REPO=${CONF_URL#*/}

PUSH_TO=git@$CONF_SITE:$CONF_REPO
until [ -n "$GET_REPO" ]
do
    echo -n "Which protocol do you want to use to access $CONF_REPO on $CONF_SITE? [git|ssh|http] (default: http) "
    read ANSWER
    ANSWER=`echo $ANSWER | tr "[:upper:]" "[:lower:]"`
    case "$ANSWER" in
        "ssh") GET_REPO=git@$CONF_SITE:$CONF_REPO ;;
        "http"|"") GET_REPO=https://$CONF_SITE/$CONF_REPO ;;
        "git") GET_REPO=git://$CONF_SITE/$CONF_REPO ;;
    esac
done

$RUBY autoproj_bootstrap $@ git $GET_REPO push_to=$PUSH_TO $BOOTSTRAP_ARGS

if test "x$@" != "xlocaldev"; then
    $SHELL -c '. $PWD/env.sh; autoproj update; autoproj osdeps; autoproj build'
fi

