import sys
import json
import queue
import threading
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from engine import *
from players import *
from dqn import *
from minmax import *

class Application(QApplication):
    def __init__(self):
        super().__init__(sys.argv)
        self.loop   = GameEventLoop(self.render_game)
        self.thread = None
        self.game   = None

        with open('config.json', 'r') as file:
            self.config = json.loads(file.read())

        dqn = self.config["DQN"]["target"]
        self.ai = ReversiAI( **self.config["DQN"]["list"][dqn] )
        self.ai.agent.load( 'DQN/{}'.format(dqn) )

        self._connect_widget()

    def _connect_widget(self):
        self.window = MainWindow(self)

    def __call__(self):
        self.window.show()
        sys.exit(self.exec_())

    def render_game(self):
        self.window.board.render_board()

    def tell_win_or_lose(self, color, result):
        def popup_message():
            msg = ResultMessage(color, result)
            msg()

        self.loop.queue.put(popup_message)

    def generate_human(self):
        return Human(self.tell_win_or_lose)

class MainWindow(QWidget):
    def __init__(self, app):
        super().__init__()
        self._connect_widget(app)
        self._setup_style()

    def _connect_widget(self, app):
        self.board = BoardBase(app)
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.board)
        hbox.addStretch(1)

        self.menu = Menu(app)
        vbox = QVBoxLayout()
        vbox.addWidget(self.menu)
        vbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addStretch(1)
        self.setLayout(vbox)

    def _setup_style(self):
        self.setWindowTitle("Reversi")

class BoardBase(QFrame):
    def __init__(self, app):
        super().__init__()
        self._connect_widget(app)
        self._setup_style()

    def _connect_widget(self, app):
        self._app = app

        self.square = [ [
                    Square(app, row, col)
                for col in range(8) ]
            for row in range(8) ]

        grid = QGridLayout()
        for square_row in self.square:
            for square in square_row:
                grid.addWidget(square, *square.place)
        self.setLayout(grid)

    def _setup_style(self):
        self.setStyleSheet("background-color: black;")

    def render_board(self):
        for state_row, square_row in zip( self._app.game.board.array[0:8], self.square ):
            for state, square in zip( state_row[0:8], square_row ):
                square.disk.set_state(state)

class Square(QFrame):
    def __init__(self, app, row, col):
        super().__init__()
        self.place = (row, col)
        self._connect_widget(app)
        self._setup_style()

    def _connect_widget(self, app):
        self._app = app
        self.disk = DiskGraphic(self)

    def _setup_style(self):
        size = self.disk.size + 2 * self.disk.margin
        self.setGeometry(0,0, size,size)
        self.setMinimumHeight(size)
        self.setMinimumWidth(size)
        self.setStyleSheet("background-color: green;")

    def mousePressEvent(self, event):
        if self._app.game is not None:
            turn = self._app.game.board.current_turn
            player = self._app.game.player[turn]
            if hasattr(player, 'tell_where_clicked'):
                self._app.thread = GameThread(target=player.tell_where_clicked, args=self.place)
                self._app.thread.start()

class DiskGraphic(QFrame):
    size = 50
    margin = 5

    def __init__(self, parent):
        super().__init__(parent)
        self._setup_style()

    def _setup_style(self):
        self.setGeometry(self.margin,self.margin, self.size,self.size)
        self.setMinimumHeight(self.size)
        self.setMinimumWidth(self.size)
        self.set_state(Disk.null)

    def set_state(self, state):
        if state is Disk.dark:
            color = "black"
        elif state is Disk.light:
            color = "white"
        else:
            color = "green"

        self.setStyleSheet(
            "background-color: {};".format(color) +
            "border-radius: {}px;".format(self.size/2) )

class Menu(QFrame):
    def __init__(self, app):
        super().__init__()
        self._connect_widget(app)

    def _connect_widget(self, app):
        self.player_choice = {
            Disk.dark : PlayerChoice(app),
            Disk.light: PlayerChoice(app) }
        self.new_game_button = NewGameButton(app)

        hbox = QHBoxLayout()
        hbox.addWidget(self.player_choice[Disk.dark])
        hbox.addWidget(self.player_choice[Disk.light])
        hbox.addWidget(self.new_game_button)
        self.setLayout(hbox)

class PlayerChoice(QComboBox):
    def __init__(self, app):
        super().__init__()
        self._connect_widget(app)
        self._setup_style()

    def _connect_widget(self, app):
        self._app = app

    def _setup_style(self):
        self.addItem(     "Human", self._app.generate_human )
        self.addItem(    "Random", delay(Random) )
        self.addItem(       "DQN", delay(self._app.ai.generate_player) )
        self.addItem( "MinMaxDQN", MinMaxPlayer.generate_player(self._app.ai.get_q_vals) )

    def get_player_type(self):
        return self.itemData(self.currentIndex())

class NewGameButton(QPushButton):
    def __init__(self, app):
        super().__init__("New Game")
        self._connect_widget(app)
        self._setup_style()

    def _connect_widget(self, app):
        self._app = app

    def _setup_style(self):
        self.clicked.connect(self._create_new_game)

    def _create_new_game(self):
        if self._app.thread is not None and self._app.thread.is_alive():
            self._app.game.need_game_stop = True
            self._app.thread.join()

        player_type = ( self._app.window.menu.player_choice[color].get_player_type() for color in (Disk.dark, Disk.light) )
        self._app.game = Reversi(
            *( ptype() for ptype in player_type ),
            render = self._app.loop.emit_render_event )

        self._app.thread = GameThread(target=self._app.game.start_game)
        self._app.thread.start()

class ResultMessage(QMessageBox):
    def __init__(self, color, result):
        super().__init__()
        self._setup_style(color, result)

    def __call__(self):
        self.exec_()

    def _setup_style(self, color, result):
        self.setWindowTitle("Game Result")
        self.setText("{} Player {}!".format(str(color), str(result)))

class GameThread(threading.Thread):
    def __init__(self, target, args=()):
        super().__init__(target=target, args=args)
        self.daemon = True

class GameEventLoop(QTimer):
    def __init__(self, render):
        super().__init__()
        self.queue = queue.Queue()
        self._render = render
        self._need_render = False
        self.timeout.connect(self._run)
        self.start(100)

    def _run(self):
        if self._need_render:
            self._need_render = False
            self._render()

        while not self.queue.empty():
            func = self.queue.get_nowait()
            func()

    def emit_render_event(self):
        self._need_render = True
